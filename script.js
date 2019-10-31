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
	
	checkParameters() {
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
		this.updateDisplay();
		return true;
	}
	
	createIndividual() {
		const cnn = new Cnn({
			kernelSize: getRandomInt(3) + 3,
			filters: getRandomInt(5) + 5,
			strides: getRandomInt(2) + 1,
			activation: 'relu',
			kernelInitializer: 'varianceScaling',
			poolSize: getRandomInt(3) + 1,
			poolStrides : getRandomInt(3) + 1 ,
			denseActivation: 'softmax',
			loss: 'categoricalCrossentropy',
			optimizer: tf.train.adam()
		}, this.epoch, 'Generation');
		return cnn;
	}
	
	breedParents(parent1, parent2) {
		var child1 = {};
		var child2 = {};
		if(getRandomInt(2) == 0) {
			child1.kernelSize = parent1.kernelSize;
			child2.kernelSize = parent2.kernelSize;
		} else {
			child1.kernelSize = parent2.kernelSize;
			child2.kernelSize = parent1.kernelSize;
		}
		if(getRandomInt(2) == 0) {
			child1.filters = parent1.filters;
			child2.filters = parent2.filters;
		} else {
			child1.filters = parent2.filters;
			child2.filters = parent1.filters;
		}
		var result = new Array();
		result.push(new Cnn(child1, this.epoch, 'Breed'));
		result.push(new Cnn(child2, this.epoch, 'Breed'));
		return result;
	}
	
	breed() {
		var newPopulation = new Array();
		for(var i = 0; i < this.breedOptions.keepTops; i = i + 1) {
			newPopulation.push(this.population[i]);
		}
		var numberToCreateBreed = Math.round(this.popSize * this.breedOptions.breedRate) / 2;
		for(var i = 0; i < numberToCreateBreed; i = i + 1) {
			// Randomly select 2 individuals from top list
			var parent1 = this.population[getRandomInt(numberToCreateBreed)];
			var parent2 = this.population[getRandomInt(numberToCreateBreed)];
			newPopulation.push(this.breedParents(parent1, parent2));
		}
		for(var i = newPopulation.length; i < this.popSize; i = i + 1) {
			newPopulation.push(new this.createIndividual());
		}
		this.population = newPopulation;
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
		
		var best = this.population[0];
		if(best.score != null) {
			$("#best-epoch").text("Created at Epoch " + best.epoch);
			if(best.creationMethod == 'Breed') {
				$("#best-generation-type").text("Created by breeding");
			} else {
				$("#best-generation-type").text("Randomly generated");
			}
			$("#best-score").text("Score: " + best.score);
		}
		
	}
}


async function runRaw() {
	var totalScore = 0.0;
	for(var i = 0; i < 3; i = i + 1) {
		const data = new MnistData();
		await data.load();
		const cnn = new Cnn({});
		const model = cnn.getModel();
		tfvis.show.modelSummary({
			name: 'Model Architecture'
		}, model);

		await cnn.train(model, data);
		await cnn.showAccuracy(model, data);
		totalScore += cnn.score;
		console.log("Total score at run " + (i+1) + ": " + totalScore);
	}
	totalScore = totalScore / 3.0;
	console.log("Average score: " + totalScore);
}

//document.addEventListener('DOMContentLoaded', runRaw);

async function run() {
	$("#launchBtn").attr("disabled", "disabled");
	var genetic = new Genetic($("#setup-population-size").val(), $("#setup-max-round").val(), {});
	await genetic.run();	
	$("#launchBtn").removeAttr("disabled");
}

$("#launchBtn").on("click", function() {
	$("#main-run").css("display", "flex");
	$("#main-idle").css("display", "none");
	run();
});

$("#launchBaseline").on("click", function() {
	runRaw();
});