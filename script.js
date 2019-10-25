import {MnistData, Cnn} from './script_mnist.js';


/**************************************************/
/********************** Run ***********************/
/**************************************************/

function getRandomInt(max) {
	return Math.floor(Math.random() * Math.floor(max));
}

class Genetic {
	constructor(popSize, epochs) {
		this.popSize = popSize;
		this.epochs = epochs;
		this.population = new Array();
		this.trained = 0;
	}
	
	createInitialPop() {
		for(var i = 0; i < this.popSize; i = i + 1) {
			this.population.push(this.createIndividual());
		}
	}
	
	train() {
		this.trained = 0;
		var $this = this;
		var i = 1;
		this.population.forEach(function(cnn) {
			if(cnn.score == null) {
				console.log("Training individual " + i++);
				$this.trainIndividual(cnn);
			} else {
				console.log("Skipping individual " + i++);
				$this.trained += 1;
			}
		});
	}
	
	createIndividual() {
		const cnn = new Cnn({
			kernelSize: getRandomInt(3) + 3,
			filters: getRandomInt(5) + 5,
			strides: 1,
			activation: 'relu',
			kernelInitializer: 'varianceScaling',
			poolSize: 2,
			poolStrides : 2,
			denseActivation: 'softmax',
			loss: 'categoricalCrossentropy',
			optimizer: tf.train.adam()
		});
		return cnn;
	}
	
	async trainIndividual(cnn) {
		const data = new MnistData();
		await data.load();
		const model = cnn.getModel();
		await cnn.train(model, data);
		await cnn.showAccuracy(model, data);
		this.trained += 1;
		console.log("Number of trained: " + this.trained);
		console.log("Training complete: " + cnn.score);
	}
}


async function runRaw() {
    const data = new MnistData();
    await data.load();
	const cnn = new Cnn({
		kernelSize: 5
	});
    const model = cnn.getModel();
    tfvis.show.modelSummary({
        name: 'Model Architecture'
    }, model);

    await cnn.train(model, data);
    await cnn.showAccuracy(model, data);
	console.log("Training complete: " + cnn.score);
}

//document.addEventListener('DOMContentLoaded', runRaw);

async function run() {
	var genetic = new Genetic($("#setup-population-size").val(), $("#setup-max-round").val());
	console.log("Create initial population");
	genetic.createInitialPop();
	console.log("Population created - " + genetic.population.length + " individuals");
	genetic.train();
	
}

$("#launchBtn").on("click", function() {
	run();
});