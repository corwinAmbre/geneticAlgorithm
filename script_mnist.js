const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const TRAIN_TEST_RATIO = 5 / 6;

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
	
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }

    async load() {
        // Make a request for the MNIST sprited image.
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const imgRequest = new Promise((resolve, reject) => {
                img.crossOrigin = '';
                img.onload = () => {
                    img.width = img.naturalWidth;
                    img.height = img.naturalHeight;

                    const datasetBytesBuffer =
                        new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

                    const chunkSize = 5000;
                    canvas.width = img.width;
                    canvas.height = chunkSize;

                    for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                        const datasetBytesView = new Float32Array(
                                datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                                IMAGE_SIZE * chunkSize);
                        ctx.drawImage(
                            img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                            chunkSize);

                        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                        for (let j = 0; j < imageData.data.length / 4; j++) {
                            // All channels hold an equal value since the image is grayscale, so
                            // just read the red channel.
                            datasetBytesView[j] = imageData.data[j * 4] / 255;
                        }
                    }
                    this.datasetImages = new Float32Array(datasetBytesBuffer);

                    resolve();
                };
                img.src = MNIST_IMAGES_SPRITE_PATH;
            });

        const labelsRequest = fetch(MNIST_LABELS_PATH);
        const[imgResponse, labelsResponse] =
            await Promise.all([imgRequest, labelsRequest]);

        this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

        // Slice the the images and labels into train and test sets.
        this.trainImages =
            this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.trainLabels =
            this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
        this.testLabels =
            this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels], () => {
            this.shuffledTrainIndex =
                (this.shuffledTrainIndex + 1) % this.trainIndices.length;
            return this.trainIndices[this.shuffledTrainIndex];
        });
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
            this.shuffledTestIndex =
                (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        });
    }

    nextBatch(batchSize, data, index) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();

            const image =
                data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label =
                data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

        return {
            xs,
            labels
        };
    }
}

/**************************************************/
/********************** CNN ***********************/
/**************************************************/

/**
 * A class that creates and train a CNN based on parameters provided in constructor
 *
 */

export class Cnn {
    constructor(modelSetup, epoch, creationMethod) {
		this.setup = Object.assign({
			kernelSize: 5,
			filters: 8,
			strides: 1,
			activation: 'relu',
			kernelInitializer: 'varianceScaling',
			poolSize: 2,
			poolStrides : 2,
			denseActivation: 'softmax',
			loss: 'categoricalCrossentropy',
			optimizer: tf.train.adam()
		}, modelSetup);
		this.score = null;
		this.epoch = epoch;
		this.creationMethod = creationMethod;
	}

    getModel() {
        const model = tf.sequential();

        const IMAGE_WIDTH = 28;
        const IMAGE_HEIGHT = 28;
        const IMAGE_CHANNELS = 1;

        // In the first layer of our convolutional neural network we have
        // to specify the input shape. Then we specify some parameters for
        // the convolution operation that takes place in this layer.
        model.add(tf.layers.conv2d({
                inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
                kernelSize: this.setup.kernelSize,
                filters: this.setup.filters,
                strides: this.setup.strides,
                activation: this.setup.activation,
                kernelInitializer: this.setup.kernelInitializer,
            }));

        // The MaxPooling layer acts as a sort of downsampling using max values
        // in a region instead of averaging.
        model.add(tf.layers.maxPooling2d({
                poolSize: [this.setup.poolSize, this.setup.poolSize],
                strides: [this.setup.poolStrides, this.setup.poolStrides]
            }));

        // Repeat another conv2d + maxPooling stack.
        // Note that we have more filters in the convolution.
        model.add(tf.layers.conv2d({
                kernelSize: this.setup.kernelSize,
                filters: this.setup.filters,
                strides: this.setup.strides,
                activation: this.setup.activation,
                kernelInitializer: this.setup.kernelInitializer
            }));
        model.add(tf.layers.maxPooling2d({
                poolSize: [this.setup.poolSize, this.setup.poolSize],
                strides: [this.setup.poolStrides, this.setup.poolStrides]
            }));

        // Now we flatten the output from the 2D filters into a 1D vector to prepare
        // it for input into our last layer. This is common practice when feeding
        // higher dimensional data to a final classification output layer.
        model.add(tf.layers.flatten());

        // Our last layer is a dense layer which has 10 output units, one for each
        // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
        const NUM_OUTPUT_CLASSES = 10;
        model.add(tf.layers.dense({
                units: NUM_OUTPUT_CLASSES,
                kernelInitializer: this.setup.kernelInitializer,
                activation: this.setup.denseActivation
            }));

        // Choose an optimizer, loss function and accuracy metric,
        // then compile and return the model
        const optimizer = this.setup.optimizer;
        model.compile({
            optimizer: optimizer,
            loss: this.setup.loss,
            metrics: ['accuracy'],
        });

        return model;
    }

    async train(model, data) {
        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
        const container = {
            name: 'Model Training',
            styles: {
                height: '1000px'
            }
        };
        //const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

        const BATCH_SIZE = 512;
        const TRAIN_DATA_SIZE = 5500;
        const TEST_DATA_SIZE = 1000;

        const[trainXs, trainYs] = tf.tidy(() => {
                const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
                return [
                    d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
                    d.labels
                ];
            });

        const[testXs, testYs] = tf.tidy(() => {
                const d = data.nextTestBatch(TEST_DATA_SIZE);
                return [
                    d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
                    d.labels
                ];
            });

        return model.fit(trainXs, trainYs, {
            batchSize: BATCH_SIZE,
            validationData: [testXs, testYs],
            epochs: 10,
            shuffle: true,
			callbacks: null
            //callbacks: fitCallbacks
        });
    }

    

    doPrediction(model, data, testDataSize = 500) {
        const IMAGE_WIDTH = 28;
        const IMAGE_HEIGHT = 28;
        const testData = data.nextTestBatch(testDataSize);
        const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
        const labels = testData.labels.argMax([-1]);
        const preds = model.predict(testxs).argMax([-1]);

        testxs.dispose();
        return [preds, labels];
    }

    async showAccuracy(model, data) {
        const[preds, labels] = this.doPrediction(model, data);
        const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
		this.score = 0.0;
		var $this = this;
		classAccuracy.forEach(function(item) {
			$this.score += (1.0 - item.accuracy)**2;
		});
        const container = {
            name: 'Accuracy',
            tab: 'Evaluation'
        };
        //tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
		

        labels.dispose();
    }
}
