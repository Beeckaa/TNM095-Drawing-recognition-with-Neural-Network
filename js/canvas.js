// Categories
var categories = ['Bird', 'Sheep', 'Seaturtle', 'Hedgehog', 'Octopus', 'Giraffe', 'Cat', 'Fish', 'Butterfly', 'Lion'];

// Variables related to the canvas drawing
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var radius = 10;
var start = 0;
var end = Math.PI * 2;
var dragging = false;

canvas.width = 400;
canvas.height = 400;
context.lineWidth = radius * 2;
var rect = canvas.getBoundingClientRect();

// Fill canvas with white background color
context.fillStyle = '#fff';
context.fillRect(0, 0, canvas.width, canvas.height);

// Keep track of min/max for each axis
var minX = 400,
	minY = 400,
	maxX = 0,
	maxY = 0;

// Variables related to the network and its predictions
let label;
let net;
var predictions = new Array(categories.length).fill(0);

/* 
 * Async function to load our neural network.
 */
async function app() {
    console.log('Loading model..');

    // Load the model.
    net = await tf.loadLayersModel('http://localhost:8080/model.json');
	console.log('Successfully loaded model');

	predictionTable(predictions);
}

/* 
 * Function to create table of predictions. 
 */
var predictionTable = function(predictions) {
	document.getElementById('predictionTable').innerHTML = " ";
	var myTableDiv = document.getElementById("predictionTable");
	console.log(predictions);
	var table = document.createElement('table');
	table.border = '1';

	var tableBody = document.createElement('tbody');
	table.appendChild(tableBody);

	for (var i = 0; i < categories.length; i++) {
			var tr = document.createElement('tr');
			tableBody.appendChild(tr);

			for (var j = 0; j < 2; j++) {
				var td = document.createElement('td');
				td.width = '75';
				if (j == 0) {
					td.appendChild(document.createTextNode(categories[i]));
					tr.appendChild(td);
				} else {
					td.appendChild(document.createTextNode(Math.round(predictions[i]*100)+ "%"));
					tr.appendChild(td);
				}
		}
	}
	myTableDiv.appendChild(table);
}

/* 
 * Function to clear the canvas. 
 */
var clearCanvas = function() {
	// Set the variables to its initial state 
	// and add a new white background to the canvas.
	minX = 400;
	minY = 400;
	maxX = 0;
	maxY = 0;
    context.clearRect(0, 0, canvas.width, canvas.height);

	context.fillStyle = '#fff';
	context.fillRect(0,0,canvas.width, canvas.height);
	document.getElementById('speech-bubble').innerHTML = "Hi, draw something on the canvas!";
	document.getElementById('predictionTable').innerHTML = " ";
	document.getElementById('zoo-keeper').src='images/zoo-keeper2.svg';
	predictions = new Array(categories.length).fill(0);
	predictionTable(predictions);
}

/* 
 * Function to draw on the canvas (listens to mousemove eventlistener). 
 */
var drawing = function(e){
	if(dragging) {
		context.strokeStyle = '#000';
		context.fillStyle = '#000';
		context.lineTo(e.offsetX, e.offsetY);
		context.stroke();
		context.beginPath();
		context.arc(e.offsetX, e.offsetY, radius, start, end);
		context.fill();
		context.beginPath();
		context.moveTo(e.offsetX, e.offsetY);

		var x = e.clientX - rect.left;
		var y = e.clientY - rect.top;
		
		// When something is drawn, calculate its impact (position and radius)
		_minX = x - radius*2 - 10;
		_minY = y - radius*2 - 10;
		_maxX = x + radius*2;
		_maxY = y + radius*2;

		// Calculate new min/max boundary
		if (_minX < minX) minX = _minX > 0 ? _minX : 0;
		if (_minY < minY) minY = _minY > 0 ? _minY : 0;
		if (_maxX > maxX) maxX = _maxX < canvas.width  ? _maxX : canvas.width;
		if (_maxY > maxY) maxY = _maxY < canvas.height ? _maxY : canvas.height;
	}
}

/* 
 * Function to start draw on canvas (listens to mousedown eventlistener). 
 */
var startDraw = function(e){
	dragging = true;
	drawing(e);
}

/* 
 * Function to stop draw on canvas (listens to mouseup eventlistener). 
 */
var stopDraw = function(){
	dragging = false;
	context.beginPath();
	context.save()
	predict();
}

/* 
 * Async function to make the prediction from our canvas. 
 */
async function predict() { 

	// Get the image data from the canvas with boundingbox
    const imgData = getImageData();

	// Draw imgData on canvas (remove this later)
	//context.fillStyle = '#000';
	//context.fillRect(0, 0, canvas.width, canvas.height);
	//context.putImageData(imgData, 100, 100);

	// Do the prediction and preprocess the canvas
	var predictions = await net.predict(preprocessCanvas(imgData));
	console.log("Predictions: " + predictions);

	a = predictions.dataSync();
	arr = Array.from(a);

	predictionTable(arr);
	console.log(arr)

	label = arr.indexOf(Math.max(...arr));
	var predictionText = '';
	var percent = '';

	switch(label) {
		case 0:
			predictionText = categories[0];
			percent = arr[0];
			console.log('Bird')
			break;
		case 1:
			predictionText = categories[1];
			percent = arr[1];
			console.log('Sheep')
			break;
		case 2:
			predictionText = categories[2];
			percent = arr[2];
			console.log('turtle')
			break;
		case 3:
			predictionText = categories[3];
			percent = arr[3];
			console.log('Hedgehog')
			break;
		case 4:
			predictionText = categories[4];
			percent = arr[4];
			console.log('Octopus')
			break;
		case 5:
			predictionText = categories[5];
			percent = arr[5];
			console.log('Giraffe')
			break;
		case 6:
			predictionText = categories[6];
			percent = arr[6];
			console.log('Cat')
			break;
		case 7:
			predictionText = categories[7];
			percent = arr[7];
			console.log('Fish')
			break;
		case 8:
			predictionText = categories[8];
			percent = arr[8];
			console.log('Butterfly')
			break;
		case 9:
			predictionText = categories[9];
			percent = arr[9];
			console.log('Lion')
			break;
		default:
			break;
	}
 
	// Text messages to be shown in speech-bubble and change AI-robot image
	if (percent*100 <= 30) {
		document.getElementById('speech-bubble').innerHTML = "You're really bad at drawing!";
		document.getElementById('zoo-keeper').src='images/zoo-keeper5.svg';
	} else if (percent*100 <= 50 && percent*100 >= 30) {
		document.getElementById('speech-bubble').innerHTML = "Hmm... that doesn't look like anything!";
		document.getElementById('zoo-keeper').src='images/zoo-keeper5.svg';
	} else if (percent*100 <= 70 && percent*100 >= 50) {
		document.getElementById('speech-bubble').innerHTML = "Draw some more...";
		document.getElementById('zoo-keeper').src='images/zoo-keeper3.svg';
	} else if (percent*100 <= 90 && percent*100 >= 70) {
		document.getElementById('speech-bubble').innerHTML = "I'm " + Math.round(percent*100) + "% sure your drawing is a " + predictionText + "!";
		document.getElementById('zoo-keeper').src='images/zoo-keeper4.svg';
	} else {
		document.getElementById('speech-bubble').innerHTML = "Oh, I know! I'm sure your drawing is a " + predictionText + "!";
		document.getElementById('zoo-keeper').src='images/zoo-keeper4.svg';
	}
}

/* 
 * Eventlisteners. 
 */
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', drawing);
canvas.addEventListener('mouseup', stopDraw);

/* 
 * Get the position of a touch relative to the canvas.
 */
function getTouchPos(canvasDom, touchEvent) {
  var rect = canvasDom.getBoundingClientRect();
  console.log("Rect " + rect.left + " " + rect.top);
  return {
    x: touchEvent.touches[0].clientX - rect.left,
    y: touchEvent.touches[0].clientY - rect.top
  };
}

/* 
 * Eventlistener and function for touchstart. 
 */
canvas.addEventListener('touchstart', function(e){
    dragging = true;
    e.preventDefault();
	mousePos = getTouchPos(canvas, e);
	var touch = e.touches[0];
	var mouseEvent = new MouseEvent("mousedown", {
		clientX: touch.clientX,
		clientY: touch.clientY
	});
	canvas.dispatchEvent(mouseEvent);
}, false);

/* 
 * Eventlistener and function for touchmove. 
 */
canvas.addEventListener('touchmove', function(e){
    var touchobj = e.changedTouches[0] // Reference first touch point for this event
	console.log(touchobj);
    if(dragging) {

		var mouseEvent = new MouseEvent("mousemove", {
			clientX: touchobj.clientX,
			clientY: touchobj.clientY
		});
		canvas.dispatchEvent(mouseEvent);

		var x = touchobj.clientX - rect.left;
		var y = touchobj.clientY - rect.top;
		
		// When something is drawn, calculate its impact (position and radius)
		_minX = x - radius;
		_minY = y - radius;
		_maxX = x + radius;
		_maxY = y + radius;

		// Calculate new min/max boundary
		if (_minX < minX) minX = _minX > 0 ? _minX : 0;
		if (_minY < minY) minY = _minY > 0 ? _minY : 0;
		if (_maxX > maxX) maxX = _maxX < canvas.width  ? _maxX : canvas.width;
		if (_maxY > maxY) maxY = _maxY < canvas.height ? _maxY : canvas.height;
	}
    e.preventDefault();
}, false);

/* 
 * Eventlistener and function for touchend. 
 */
canvas.addEventListener('touchend', function(e){
    dragging = false;
	context.beginPath();
    e.preventDefault();
	var mouseEvent = new MouseEvent("mouseup", {});
  	canvas.dispatchEvent(mouseEvent);
	predict();
}, false);

/* 
 * Preprocessing of the canvas make it into a tensor to fit the model. 
 */
preprocessCanvas = (drawing) => {
	// Preprocess image for the network
	let tensor = tf
	.browser.fromPixels(drawing) // Shape: (400, 400, 3) - RGB image
	.resizeNearestNeighbor([28, 28]) // Shape: (28, 28, 3) - RGB image
	.mean(2) // Shape: (28, 28) - grayscale
	.expandDims(2) // Shape: (28, 28, 1) - network expects 3d values with channels in the last dimension
	.expandDims() // Shape: (1, 784) - network makes predictions for "batches" of images
	.toFloat(); // Network works with floating points inputs
	return tensor.div(255.0); // Normalize [0..255] values into [0..1] range

	/*return tf.tidy(()=>{
        let tensor = tf.browser.fromPixels(drawing, 1); // convert the image data to a tensor

        // resize to 28 x 28  
        const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat(); 

        // Normalize the image 
        const offset = tf.scalar(255.0);
        const normalized = tf.scalar(1.0).sub(resized.div(offset));

        // insert a dimension of 1 into a tensor's shape
        const batched = normalized.expandDims(0);
        return batched;
    })*/
}

/* 
 * Function to get the current image data.
 */
function getImageData() {
	// Get the minimum bounding box around the drawing (with equal sides)
	if (maxX > maxY) {
		maxY = maxX;
	} else {
		maxX = maxY;
	}
	const imgData = context.getImageData(minX, minY, maxX, maxY);

	// Get the image data according to DPI
	//const dpi = window.devicePixelRatio;
	//const imgData = context.getImageData(minX * dpi, minY * dpi, (maxX - minX) * dpi, (maxY - minY) * dpi);

	var dataArr = imgData.data;

	// Inverting the colors in imgData.
	for(var i = 0; i < dataArr.length; i += 4) {
		var r = dataArr[i]; 	// Red color lies between 0 and 255
		var g = dataArr[i + 1]; // Green color lies between 0 and 255
		var b = dataArr[i + 2]; // Blue color lies between 0 and 255
		var a = dataArr[i + 3]; // Transparency lies between 0 and 255

		var invertedRed = 255 - r;
		var invertedGreen = 255 - g;
		var invertedBlue = 255 - b;

		dataArr[i] = invertedRed;
		dataArr[i + 1] = invertedGreen;
		dataArr[i + 2] = invertedBlue;
	}

	return imgData;
}

