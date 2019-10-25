import {MnistData, Cnn} from './script_mnist.js';


/**************************************************/
/********************** Run ***********************/
/**************************************************/

function getRandomInt(max) {
	return Math.floor(Math.random() * Math.floor(max));
}

class Genetic {
	constructor(popSize, epochs, breedOptions) {
		this.popSize = popSize;
		this.epochs = epochs;
		this.population = new Array();
		this.trained = 0;
		this.trainOnGoing = 0;
		this.epoch = 0;
		this.breedOptions = Object.assign({
			keepTops: 1,
			breedRate: 0.4
		}, breedOptions);
	}
	
	async run() {
		this.createInitialPop();
		await this.train();
		for(var i = 1; i < this.epochs; i = i + 1) {
			this.breed();
			await this.train();
		}
	}
	
	createInitialPop() {
		for(var i = 0; i < this.popSize; i = i + 1) {
			this.population.push(this.createIndividual());
		}
	}
	
	async train() {
		this.epoch += 1;
		this.trained = 0;
		this.trainOnGoing = 0;
		this.updateDisplay();
		var promises = new Array();
		for(var i = 0; i < this.popSize; i = i + 1) {
			var cnn = this.population[i];
			if(cnn.score == null) {
				console.log("Training individual " + (i+1));
				promises.push(this.trainIndividual(cnn));
			} else {
				console.log("Skipping individual " + (i+1));
				this.trained += 1;
				this.updateDisplay();
			}
		}
		await Promise.all(promises);
		this.population.sort(function(a, b) {
			return a.score - b.score;
		});
		return true;
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
	
	breed() {
	}
	
	async trainIndividual(cnn) {
		this.trainOnGoing += 1;
		this.updateDisplay();
		const data = new MnistData();
		await data.load();
		const model = cnn.getModel();
		await cnn.train(model, data);
		await cnn.showAccuracy(model, data);
		this.trained += 1;
		this.trainOnGoing -= 1;
		this.updateDisplay();
		console.log("Training complete: " + cnn.score);
	}
	
	updateDisplay() {
		$("#genetic-epoch").text(this.epoch);
		var notStarted = this.popSize - this.trainOnGoing - this.trained;
		$("#genetic-to-train").text(notStarted == 0 ? '' : notStarted);
		$("#genetic-train-on-going").text(this.trainOnGoing == 0 ? '' : this.trainOnGoing);
		$("#genetic-trained").text(this.trained == 0 ? '' : this.trained);
		$("#genetic-to-train").css("witdh", notStarted * 400 / this.popSize + "px");
		$("#genetic-train-on-going").css("width", this.trainOnGoing * 400 / this.popSize + "px");
		$("#genetic-trained").css("width", this.trained * 400 / this.popSize + "px");
		
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
	var genetic = new Genetic($("#setup-population-size").val(), $("#setup-max-round").val(), {});
	genetic.run();	
}

$("#launchBtn").on("click", function() {
	$("#main-run").css("display", "flex");
	$("#main-idle").css("display", "none");
	run();
});