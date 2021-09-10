
import datetime, os
import tensorflow as tf

dataset_pattern = "../Datasets/safebooru_r63_256/train/female/*"
example_image_path = "../Datasets/safebooru_r63_256/train/female/"\
"00af2f4796bcf58f445ab78e4f8a42f4931c28eec024de0e79872fa019575c5f.png"

size = 256
pixel_size = 128
block_depth = 2

batch_size = 8
steps_per_epoch = 1000
epochs = 1000

steps = 20

residual = False
concat = True

mixed_precision = False

def log_noise_schedule(r):
    # returns standard deviation at step t/T
    return tf.math.log(1.0 + (1.0 / 256 - 1.0) * r)

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

optimizer = tf.keras.optimizers.Adam(1e-5)

if mixed_precision:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

class Residual(tf.keras.layers.Layer):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def build(self, input_shape):
        self.module.build(input_shape)
        if residual or concat:
            self.dense = tf.keras.layers.Dense(input_shape[-1], use_bias=False)

    def call(self, input):
        if residual:
            return input + self.dense(self.module(input))
        elif concat:
            return self.dense(tf.concat([self.module(input), input], -1))
        else:
            return self.module(input)

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, dilation = 1):
        super().__init__()

        self.filters = filters
        self.dilation = dilation

    def build(self, input_shape):
        self.module = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.filters, 3, 1, 'same', activation='relu', 
                dilation_rate=(self.dilation, self.dilation)
            ) for i in range(block_depth)
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
            self.middle = Residual(
                tf.keras.Sequential([
                    DownShuffle(),
                    Block(pixel_size),
                    self.middle, 
                    Block(pixel_size),
                    UpShuffle(),
                ])
            )
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
        log_scale = log_noise_schedule(
            tf.random.uniform(tf.shape(input)[:-3])
        )[..., None, None, None]
        scale = tf.exp(log_scale)
        epsilon = tf.random.normal(tf.shape(input))

        noised = (
            input * tf.sqrt(1 - tf.square(scale)) + 
            epsilon * scale
        )

        fake = self.denoiser((noised, log_scale))
        return tf.math.squared_difference(fake, input)

denoiser = Denoiser()
trainer = Trainer(denoiser)

def load_file(file, crop=True):
    image = tf.image.decode_jpeg(tf.io.read_file(file))[:, :, :3]
    if crop:
        image = tf.image.random_crop(image, [size, size, 3])
    image = tf.cast(image, tf.float32) / 128 - 1
    return image, image

classes = [
    dataset_pattern,
]

datasets = []

example_image = load_file(example_image_path)
example = tf.random.normal((steps, 4, size, size, 3))

for folder in classes:
    dataset = tf.data.Dataset.list_files(folder)
    dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.map(load_file).batch(batch_size).prefetch(8)
    datasets += [dataset]

@tf.function
def log_sample(epochs, logs):
    with summary_writer.as_default():
        identity = denoiser((
            example_image[0][None], 
            tf.math.log(1.0 / 256)[None, None, None, None]
        ))
        tf.summary.image('identity', identity * 0.5 + 0.5, epochs)
        del identity

        fake = example[0, ...]
        log_scale = tf.math.log(1.0)
        for i in range(steps):
            log_scale = log_noise_schedule(i / steps)

            epsilon = example[i, ...]
            scale = tf.exp(log_scale)
            sample = (
                fake * tf.sqrt(1 - tf.square(scale)) + 
                epsilon * scale
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

if __name__ == "__main__":
    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_basepath = "logs"
    logs_path = os.path.join(logs_basepath, name)
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
    summary_writer = tf.summary.create_file_writer(logs_path)

    dataset_example = next(iter(datasets[0]))[0]
    loss = identity(
        dataset_example, trainer(dataset_example)
    )
    del loss, dataset_example

    trainer.compile(
        optimizer, 
        identity
    )

    trainer.fit(
        datasets[0],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=log_sample,
                on_train_end=lambda _: log_sample(epochs, {})
            )
        ]
    )