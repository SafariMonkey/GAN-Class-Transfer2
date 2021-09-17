
import datetime, os, shutil, stat
import tensorflow as tf
import itertools
import glob

dataset_path = "/mnt/Elements/derpi/images-postprocessed/256-on-shorter-side/png-srgb-to23-sym"

dataset_pattern = f"{dataset_path}""/{:02d}/*.*"
train_pattern, val_pattern = [], []
for i in range(100):
    d = val_pattern if i % 5 == 0 else train_pattern
    d.append(dataset_pattern.format(i))

example_image_path = f"{dataset_path}/30/1250830.png"

size = 256
pixel_size = 128
block_depth = 2

batch_size = 8
steps_per_epoch = 1000
epochs = 1000

steps = 20
scales = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.99, 1.0]

skip_mode='concat'

mixed_precision = False

def log_image_signal_schedule(r):
    # an exponentially increasing signal,
    # starting at 1px worth of information
    image_signal_strength = (size*size)**(r-1.0)
    # returns image signal strength at step t/T
    return tf.math.log(image_signal_strength)

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

optimizer = tf.keras.optimizers.Adam(1e-5)

if mixed_precision:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

class Skip(tf.keras.layers.Layer):
    def __init__(self, module, mode):
        super().__init__()

        self.module = module
        self.mode = mode

    def build(self, input_shape):
        self.module.build(input_shape)
        if self.mode in ['residual', 'concat']:
            self.dense = tf.keras.layers.Dense(input_shape[-1], use_bias=False)

    def call(self, input):
        if self.mode == 'residual':
            return input + self.dense(self.module(input))
        elif self.mode == 'concat':
            return self.dense(tf.concat([self.module(input), input], -1))
        if self.mode == 'identity':
            return self.module(input)
        else:
            raise ValueError(f"Unknown residual mode {self.mode}")

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, dilation = 1):
        super().__init__()

        self.filters = filters
        self.dilation = dilation

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            Skip(tf.keras.layers.Conv2D(
                self.filters, 3, 1, 'same', activation='relu', 
                dilation_rate=(self.dilation, self.dilation),
            ), mode=skip_mode) for i in range(block_depth)
        ])
        self.module.build(input_shape)

    def call(self, input):
        return self.module(input)

class UpShuffle(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        return tf.keras.layers.UpSampling2D(interpolation='bilinear')(input)

class DownShuffle(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        return tf.keras.layers.AveragePooling2D()(input)

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.rgb_dense = tf.keras.layers.Dense(pixel_size, use_bias=False)
        self.scale_dense = tf.keras.layers.Dense(pixel_size)
        self.output_dense = tf.keras.layers.Dense(pixel_size)

    def build(self, input_shape):
        rgb, scale = input_shape
        self.rgb_dense.build(rgb)
        self.scale_dense.build(scale)
        self.output_dense.build([pixel_size])

    def call(self, input):
        rgb, scale = input
        return self.output_dense(tf.nn.relu(
            self.rgb_dense(rgb) + 
            self.scale_dense(scale)
        ))

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        rgb, scale = tf.split(input, [3, 1], -1)
        scale = tf.reduce_mean(scale, axis=[-1, -2, -3], keepdims=True)
        return rgb, scale

@tf.function
def identity(y_true, y_pred):
    return tf.reduce_mean(y_pred)

class Denoiser(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.middle = Block(pixel_size)
        for i in range(7):
            self.middle = Skip(tf.keras.Sequential([
                DownShuffle(),
                Block(pixel_size),
                self.middle, 
                Block(pixel_size),
                UpShuffle(),
            ]), mode=skip_mode)
        self.middle = tf.keras.Sequential([
            Block(pixel_size),
            self.middle,
            Block(pixel_size),
            tf.keras.layers.Dense(pixel_size, activation='relu'),
            tf.keras.layers.Dense(3),
        ])

    def call(self, input):
        return self.middle(self.encoder(input))

class Trainer(tf.keras.Model):
    def __init__(self, denoiser):
        super().__init__()

        self.denoiser = denoiser

    def call(self, input):
        log_scale = log_image_signal_schedule(
            tf.random.uniform(tf.shape(input)[:-3])
        )[..., None, None, None]
        scale = tf.exp(log_scale)
        epsilon = tf.random.normal(tf.shape(input))

        noised = (
            input * tf.sqrt(scale) + 
            epsilon * tf.sqrt(1 - scale)
        )

        fake = self.denoiser((noised, log_scale))
        return tf.math.squared_difference(fake, input)

denoiser = Denoiser()
trainer = Trainer(denoiser)

def load_file(file, crop=True):
    image = tf.image.decode_jpeg(tf.io.read_file(file), 3)[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    image = tf.cast(image, tf.float32) / 128 - 1
    return image, image

example_image = load_file(example_image_path)
example = tf.random.normal((steps, 4, size, size, 3))
example_denoise = tf.random.normal((len(scales), 4, size, size, 3))

datasets = []
for pattern in train_pattern, val_pattern:
    dataset = tf.data.Dataset.list_files(pattern)
    dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.map(load_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

train_dataset, val_dataset = datasets

def log_sample(epochs, logs):
    _log_sample(tf.constant(epochs, dtype='int64'))

@tf.function(input_signature=(tf.TensorSpec((), dtype='int64'),))
def _log_sample(epochs):
    with summary_writer.as_default():
        for i in range(len(scales)):
            scale = scales[i]
            sample = (
                example_image[0] * tf.sqrt(scale) + 
                example_denoise[i, ...] * tf.sqrt(1 - scale)
            )
            denoised = denoiser((
                sample, 
                tf.math.log(scale)[None, None, None, None]
            ))
            tf.summary.image(f'denoise_{scale}_input', sample * 0.5 + 0.5, epochs, 4)
            tf.summary.image(f'denoise_{scale}_output', denoised * 0.5 + 0.5, epochs, 4)
            del sample, denoised

        fake = example[0, ...]
        log_scale = tf.math.log(1.0)
        for i in range(steps):
            log_scale = log_image_signal_schedule(i / steps)

            epsilon = example[i, ...]
            scale = tf.exp(log_scale)
            sample = (
                fake * tf.sqrt(scale) + 
                epsilon * tf.sqrt(1 - scale)
            )

            log_scale = log_scale[None, None, None, None]
            fake = denoiser((sample, log_scale))

            if i == 0:
                tf.summary.image('step_0', fake * 0.5 + 0.5, epochs, 4)
            if i == steps // 4:
                tf.summary.image('step_0.25', fake * 0.5 + 0.5, epochs, 4)
            if i == 2 * steps // 4:
                tf.summary.image('step_0.5', fake * 0.5 + 0.5, epochs, 4)
            if i == 3 * steps // 4:
                tf.summary.image('step_0.75', fake * 0.5 + 0.5, epochs, 4)
        tf.summary.image('fake', fake * 0.5 + 0.5, epochs, 4)
        tf.summary.image('sample', sample * 0.5 + 0.5, epochs, 4)
        dataset_examples = tf.stack(list(v[0][0] for v in itertools.islice(iter(train_dataset), 4)))
        tf.summary.image('dataset_example', dataset_examples * 0.5 + 0.5, epochs, 4)

class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, epoch_per_save=1, *args, **kwargs):
        self.epochs_per_save = epoch_per_save
        super().__init__(save_freq='epoch', *args, **kwargs)

    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % self.epochs_per_save == 0:
            super().on_epoch_end(epoch, logs)

if __name__ == "__main__":
    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_basepath = "logs"
    logs_path = os.path.join(logs_basepath, name)
    os.mkdir(logs_path)
    current_path = os.path.join(logs_basepath, "current")
    try:
        os.unlink(current_path)
    except OSError:
        pass
    os.symlink(
        os.path.relpath(
            logs_path,
            logs_basepath,
        ),
        current_path
    )
    print(f"logs: {logs_path}")
    for file in glob.glob(f"*.py", recursive=True):
        out_file = f"{logs_path}/{file}"
        shutil.copyfile(file, out_file)
        st = os.stat(file)
        all_write = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        os.chmod(out_file, st.st_mode & ~(all_write))

    summary_writer = tf.summary.create_file_writer(logs_path)

    dataset_example = next(iter(train_dataset))[0]
    loss = identity(
        dataset_example, trainer(dataset_example)
    )
    del loss, dataset_example

    trainer.compile(
        optimizer, 
        identity
    )

    trainer.summary()

    trainer.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=steps_per_epoch,
        validation_freq=5,
        callbacks=[
            EpochModelCheckpoint(
                filepath=f'{logs_path}''/best.SavedModel',
                save_best_only=True, monitor='val_loss', mode='min',
                epoch_per_save=10,
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=log_sample,
                on_train_end=lambda _: log_sample(epochs, {})
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./tensorboard-callback-logs/{name}', profile_batch=5,
            )
        ]
    )