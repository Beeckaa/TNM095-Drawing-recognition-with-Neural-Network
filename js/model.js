let net;

async function app() {
    console.log('Loading model..');

    // Load the model.
    net = await tf.loadLayersModel('http://localhost:8080/model.json');
    console.log('Successfully loaded model');

    // Make a prediction through the model on our image.
    const imgEl = document.getElementById('img');
    const result = await net.predict(imgEl);
    console.log(result);
}

app();