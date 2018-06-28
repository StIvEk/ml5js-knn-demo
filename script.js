(function () {
  let video_frame;
  let knn;

  init();

  function init() {
    //select the elements relevant to video and capture
    video_frame = document.getElementById("myVideo");
    btnCat1 = document.getElementById("btnCat1");
    btnCat2 = document.getElementById("btnCat2");
    btnCat3 = document.getElementById("btnCat3");
    btnCat4 = document.getElementById("btnCat4");
    btnCat5 = document.getElementById("btnCat5");
    btnClasify = document.getElementById("btnClasify");
    btnSave = document.getElementById("btnSave");
    btnLoad = document.getElementById("btnLoad");
    btnSaveFile = document.getElementById("btnSaveFile");
    btnLoadFile = document.getElementById("btnLoadFile");

    // obtain access to browser local system connected media
    navigator.getUserMedia = (
      //check for all available media
      //chrome
      navigator.getUserMedia ||
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia ||
      navigator.msGetUserMedia
    );


    if (navigator.getUserMedia) {
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
        btnSave.addEventListener("click", saveModel);
        btnLoad.addEventListener("click", loadModel);
        btnSaveFile.addEventListener("click", saveModelFile);
        btnLoadFile.addEventListener("click", loadModelFile);
      });

    } else {
      alert("The browser does not support Media Interface");
    }
  }

  // Capture camera image and categorize it
  function capture(knn, category) {
    console.log('Capturing...', knn, category);
    
    knn.addImageFromVideo(category.substring(4));
  }

  function categorize(event) {
    // console.log('Categorizing...', event.target.innerText);
  }

  // Clasify current camera image by predicting its category
  function clasify() {
    console.log('Recognizing...');

    knn.predictFromVideo(displayResult);
  }

  // Display predicted category
  function displayResult(results) {
    const predictionEl = document.getElementById('prediction');

    console.log(results);

    predictionEl.textContent = `Cat ${results.classIndex}`;
  }

  // Log message when model is loaded
  function modelLoaded() {
    console.log('Model loaded...');
  }

  // Save the trained model in window.localStorag
  function saveModel() {
    const c = knn.knn.classLogitsMatrices;
    const model = {
      logits: c,
      tensors: c.map((t) => t ? t.dataSync() : null)
    }

    localStorage.setItem('knnModel', JSON.stringify(model));
    
    console.log(knn.knn);
  }

  // Load trained model from window.localStorage
  function loadModel() {
    const model = JSON.parse(localStorage.getItem('knnModel'));


    // knn.knn.setClassLogitsMatrices(JSON.parse(c));

    var r = model.tensors.map(function(e, n) {
      if (e) {
        var r = Object.keys(e).map(function(t) {
          return e[t]
        });
        return ml5.dl.tensor(r, model.logits[n].shape, model.logits[n].dtype)
      }
      return null
    });
    knn.hasAnyTrainedClass = true;
    knn.knn.setClassLogitsMatrices(r);

    console.log(knn.knn);
  }

  function saveModelFile() {
    knn.save();
    
    console.log(knn.knn);
  }

  // Load trained model from window.localStorage
  function loadModelFile() {
    knn.load('1530174444835.json');

    console.log(knn.knn);
  }
})();
