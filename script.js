(function () {
  let video_frame;
  let canvas;
  let knn;

  init();

  function init() {
    //select the elements relevant to video and capture
    video_frame = document.getElementById("myVideo");
    canvas = document.getElementById("canvas");
    btnCat1 = document.getElementById("btnCat1");
    btnCat2 = document.getElementById("btnCat2");
    btnCat3 = document.getElementById("btnCat3");
    btnCat4 = document.getElementById("btnCat4");
    btnCat5 = document.getElementById("btnCat5");
    btnClasify = document.getElementById("btnClasify");


    imcanvas = canvas.getContext("2d");

    /*
    This part of javascript code will capture frames from
    the webcam and display on webpage.
    */

    //    obtain access to browser local system connected media ..

    navigator.getUserMedia = (
      //check for all available media
      //chrome
      navigator.getUserMedia ||
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia ||
      navigator.msGetUserMedia
    );

    //this will set a read-only boolean property to the obtained list of media devices

    if (navigator.getUserMedia) {
      //log ... print in the JS console in browser
      console.log("Browser supports media api");

      //specify what type of media if required.
      navigator.mediaDevices.getUserMedia({
          video: true,
          //   audio : true, //if microphone access was required
        },
        // success_stream,
        // error_stream
      ).then(function(stream) {
        console.log("Streaming successful");
        //once we have the webcam stream, we shall display it in the
        //html video element created
        // try {
        video_frame.srcObject = stream;

        knn = new ml5.KNNImageClassifier(5, 1, modelLoaded, video_frame);
        console.log('KNN: ', knn);

        //set up event listeners ..
        btnCat1.addEventListener("click", capture.bind(null, knn, 'Cat 1'));
        btnCat2.addEventListener("click", capture.bind(null, knn, 'Cat 2'));
        btnCat3.addEventListener("click", capture.bind(null, knn, 'Cat 3'));
        btnCat4.addEventListener("click", capture.bind(null, knn, 'Cat 4'));
        btnCat5.addEventListener("click", capture.bind(null, knn, 'Cat 5'));
        btnClasify.addEventListener("click", clasify);
      });

    } else {
      alert("The browser does not support Media Interface");
    }
  }

  function capture(knn, category) {
    console.log('Capturing...', knn, category);
    
    knn.addImageFromVideo(category.substring(4));
  }

  function categorize(event) {
    // console.log('Categorizing...', event.target.innerText);
  }

  function clasify() {
    console.log('Recognizing...');

    knn.predictFromVideo(displayResult);
    saveModelLocally();
  }

  function displayResult(results) {
    const predictionEl = document.getElementById('prediction');

    console.log(results);

    predictionEl.textContent = results.classIndex;
  }

  function modelLoaded() {
    console.log('Model loaded...');
  }

  function saveModelLocally() {
    const c = knn.knn.classLogitsMatrices;
    const model = {
      logits: c,
      tensors: c.map((t) => t ? t.dataSync() : null)
    }

    console.log(model);
  }
})();
