
'use strict';



const videoElement = document.querySelector('video');
const videoSelect = document.querySelector('select#videoSource');
const selectors = [videoSelect];

const canvas = document.querySelector('canvas');

let file_or_realtime = 1;

// ------------------load file-------------------------

var fileElem = document.getElementById("fileElem");
fileElem.addEventListener('change',handlefile,false);




// ------------------device setting----------------------
function gotDevices(deviceInfos) {
  // Handles being called several times to update labels. Preserve values.
  const values = selectors.map(select => select.value);
  selectors.forEach(select => {
    while (select.firstChild) {
      select.removeChild(select.firstChild);
    }
  });
  for (let i = 0; i !== deviceInfos.length; ++i) {
    const deviceInfo = deviceInfos[i];
    const option = document.createElement('option');
    option.value = deviceInfo.deviceId;
    if (deviceInfo.kind === 'videoinput') {
      option.text = deviceInfo.label || `camera ${videoSelect.length + 1}`;
      videoSelect.appendChild(option);
    } else {
      console.log('Some other kind of source/device: ', deviceInfo);
    }
  }
  selectors.forEach((select, selectorIndex) => {
    if (Array.prototype.slice.call(select.childNodes).some(n => n.value === values[selectorIndex])) {
      select.value = values[selectorIndex];
    }
  });
}

navigator.mediaDevices.enumerateDevices().then(gotDevices).catch(handleError);
// ------------------device setting----------------------


// Attach audio output device to video element using device/sink ID.
function attachSinkId(element, sinkId) {
  if (typeof element.sinkId !== 'undefined') {
    element.setSinkId(sinkId)
        .then(() => {
          console.log(`Success, audio output device attached: ${sinkId}`);
        })
        .catch(error => {
          let errorMessage = error;
          if (error.name === 'SecurityError') {
            errorMessage = `You need to use HTTPS for selecting audio output device: ${error}`;
          }
          console.error(errorMessage);
          // Jump back to first output device in the list as it's the default.
          audioOutputSelect.selectedIndex = 0;
        });
  } else {
    console.warn('Browser does not support output device selection.');
  }
}

function changeAudioDestination() {
  const audioDestination = audioOutputSelect.value;
  attachSinkId(videoElement, audioDestination);
}

function gotStream(stream) {
  window.stream = stream; // make stream available to console
  videoElement.srcObject = stream;
  // Refresh button list in case labels have become available
  return navigator.mediaDevices.enumerateDevices();
}

function handleError(error) {
  console.log('navigator.MediaDevices.getUserMedia error: ', error.message, error.name);
}

function start() {
  if (window.stream) {
    window.stream.getTracks().forEach(track => {
      track.stop();
    });
  }
  
  const videoSource = videoSelect.value;
  const constraints = {
    audio: false,
    video: {deviceId: videoSource ? {exact: videoSource} : undefined,
            "width": {
              "min": "416",
              "max": "416"
          },
          "height": {
              "min": "416",
              "max": "416"
  }},
    
  };
  navigator.mediaDevices.getUserMedia(constraints).then(gotStream).then(gotDevices).catch(handleError);
}


// ---------------- canvas ---------------------
const ctx = canvas.getContext('2d');

// ---------------- file to canvas--------------
function handlefile(e){
  console.log('handling');
  var reader = new FileReader();
  file_or_realtime = 0;
  reader.onload = function(event){
      var img = new Image();
      img.onload = function(){
          // canvas.width = img.width;
          // canvas.height = img.height;
          canvas.width = 416;
          canvas.height = 416;

          ctx.drawImage(img,0,0,416,416);
      }
      img.src = event.target.result;
  }
  reader.readAsDataURL(e.target.files[0]);     
}

function press(){
  console.log('press');
  
}


//----------- take picture ------------------
function take_picture(){
  file_or_realtime = 1;
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

}




//----------------canvas---------------

// ------------model-----------------------------------------


// ------------------- Load our model.-----------------------

var sess;
const init_model = async ()=>{
  console.log("Loading Session and model")
  sess = new onnx.InferenceSession({backendHint: 'webgl'});
  await sess.loadModel("./yolov4_tiny.onnx");
  console.log("Done")
}

init_model();




async function updatePredictions() {
  
  const imgData = ctx.getImageData(0, 0, 416, 416);//red, green, blue, alpha (0 ~ 255)
  const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");
  // console.log(input);
 
  
  const outputMap = await sess.run([input]);
  // console.log('outputMap',outputMap)
  const outputTensor = outputMap.values().next().value;
  const outputTensor2 = outputMap.values().value;
  // console.log('outputTensor',outputTensor)
  // console.log('outputTensor2',outputTensor2)
  const predictions = outputTensor.data;
  const maxPrediction = predictions.indexOf(Math.max(...predictions));
  let ans = JSON.stringify(predictions[0]);
  
  document.getElementById("ans").innerHTML = classes[maxPrediction];
  // console.log(predictions);
  let prediction = tf.tensor(predictions,[72,13,13])
  // prediction.print(true)
  // const maxPrediction = predictions.indexOf(Math.max(...predictions));
  // let ans = JSON.stringify(predictions[0])
  // document.getElementById("ans").innerHTML = classes[maxPrediction];
  // console.log('testing')

  //-----------bbox
  const input_side = 13;
  const stride_side = 32;//416/13

  const scaled_anchors = tf.tensor([2.53125, 2.5625, 4.21875, 5.28125, 10.75, 9.96875],[3,2])
  // console.log('scaled anchors')
  // scaled_anchors.print(true)
  prediction = tf.transpose(prediction.reshape([3,24,13,13]),[0,2,3,1])
  // prediction.print(true)
  const x = prediction.gather(0,3).sigmoid()
  // x.print(true)
  const y = prediction.gather(1,3).sigmoid()

  const w = prediction.gather(2,3)

  const h = prediction.gather(3,3)

  const conf = prediction.gather(4,3).sigmoid()
  let select = tf.cast(tf.linspace(5,23,19),'int32')
  
  // select.print(true)
  
  const pred_cls = prediction.gather(select,3).sigmoid();
  // pred_cls.print(true)
  const grid_x_b = tf.linspace(0,12,13).reshape([1,13]);
  let grid_x = grid_x_b
  for (let i=0;i<12;i++){
    grid_x = grid_x.concat(grid_x_b);
  }
  grid_x = grid_x.reshape([1,13,13])
  grid_x = grid_x.concat([grid_x,grid_x])
  
  // grid_x.print(true)
  let grid_y = grid_x.clone() 
  // grid_y.print(true)

  const anchor_w_b = scaled_anchors.gather(0,1).reshape([3,1])
  // anchor_w_b.print(true)
  const anchor_h_b = scaled_anchors.gather(1,1).reshape([3,1])
  // anchor_h_b.print(true)

  let anchor_w_b1 = anchor_w_b
  let anchor_h_b1 = anchor_h_b
  for(let i=0;i<12;i++){
    anchor_w_b1 = anchor_w_b1.concat(anchor_w_b,1);
    anchor_h_b1 = anchor_h_b1.concat(anchor_h_b,1);
  }
  // anchor_w_b1.print(true)
  anchor_w_b1 = anchor_w_b1.reshape([3,13,1])
  anchor_h_b1 = anchor_h_b1.reshape([3,13,1])
  let anchor_w = anchor_w_b1
  let anchor_h = anchor_h_b1
  for(let i=0;i<12;i++){
    anchor_w = anchor_w.concat(anchor_w_b1,2);
    anchor_h = anchor_h.concat(anchor_h_b1,2);
  }
  // anchor_w.print(true)
  // x.print(true)
  // grid_x.print(true)
  let pred_boxes = tf.addN([x,grid_x]).reshape([3,13,13,1])
  // pred_boxes.print(true)
  pred_boxes = pred_boxes.concat(tf.addN([y,grid_y]).reshape([3,13,13,1]),3)
  // pred_boxes.print(true)
  pred_boxes = pred_boxes.concat(tf.mul(tf.exp(w),anchor_w).reshape([3,13,13,1]),3)
  pred_boxes = pred_boxes.concat(tf.mul(tf.exp(h),anchor_h).reshape([3,13,13,1]),3)
  // pred_boxes.print(true)

  // const _scale = tf.tensor([13,13,13,13])
  // _scale.print(true)

  let output = tf.div(pred_boxes.reshape([-1,4]),tf.scalar(13))
  output = output.concat(conf.reshape([-1,1]),1)
  output = output.concat(pred_cls.reshape([-1,19]),1)
  // output.print(true)

  let box_corner = tf.addN([output.gather(0,1).reshape([-1,1]) ,output.gather(2,1).reshape([-1,1]).div(-2)]) // x1
  box_corner = box_corner.concat(tf.addN([output.gather(1,1).reshape([-1,1]) ,output.gather(3,1).reshape([-1,1]).div(-2)]),1) //y1
  box_corner = box_corner.concat(tf.addN([output.gather(0,1).reshape([-1,1]) ,output.gather(2,1).reshape([-1,1]).div(2)]),1) // x2
  box_corner = box_corner.concat(tf.addN([output.gather(1,1).reshape([-1,1]) ,output.gather(3,1).reshape([-1,1]).div(2)]),1) // y2
  
  
  // box_corner.print(true)

  
  select = tf.cast(tf.linspace(5,23,19),'int32')
  let class_conf = tf.max(output.gather(select,1),1,true)
  let class_pred = tf.argMax(output.gather(select,1),1,true).reshape([-1,1])
  // class_conf.print(true)
  // class_pred.print(true)

  //----------nms
  let for_nms_bbox = tf.addN([output.gather(1,1).reshape([-1,1]) ,output.gather(3,1).reshape([-1,1]).div(-2)]) // y1
  for_nms_bbox = for_nms_bbox.concat(tf.addN([output.gather(0,1).reshape([-1,1]) ,output.gather(2,1).reshape([-1,1]).div(-2)]),1) // x1
  for_nms_bbox = for_nms_bbox.concat(tf.addN([output.gather(1,1).reshape([-1,1]) ,output.gather(3,1).reshape([-1,1]).div(2)]),1) // y2
  for_nms_bbox = for_nms_bbox.concat(tf.addN([output.gather(0,1).reshape([-1,1]) ,output.gather(2,1).reshape([-1,1]).div(2)]),1) // x2
  
  // for_nms_bbox.print(true)
  const for_nms_conf = tf.mul(conf.reshape([-1]) ,class_conf.reshape([-1]))
  // for_nms_conf.print(true)

  const bboxes = await tf.image.nonMaxSuppressionAsync(for_nms_bbox,for_nms_conf,5,0.5,0.5)
  // bboxes.print(true)

  let ans_bbox_index = bboxes.arraySync()//.get(0)
  let pred_class = class_pred.arraySync()
  let conf_class = class_conf.arraySync()
  
  try{
    // console.log(pred_class[ans_bbox_index[0]],conf_class[ans_bbox_index[0]])
    let ans_bbox = for_nms_bbox.gather(ans_bbox_index[0],0)
    ans_bbox.print(true)
  //ans_bbox.slice([0],[2])
    let yx = tf.addN([ans_bbox.slice([0],[2]),ans_bbox.slice([2])]).div(2)
    // yx.print(true)
    let hw = tf.addN([ans_bbox.slice([0],[2]).mul(-1),ans_bbox.slice([2])])
    // hw.print(true)

    let box_min = tf.addN([yx,hw.div(-2)])
    let box_max = tf.addN([yx,hw.div(2)])
    let box_yx = box_min.mul(416)
    let box_yx2 = box_max.mul(416)
    // box_yx.print(true)
    // box_yx2.print(true)

    //------------show
    
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    // ctx.drawImage(imgData,0,0,480,480);
    let pred_box_yx = box_yx.arraySync()
    let pred_box_yx2 = box_yx2.arraySync()
    ctx.strokeStyle='red';
  
  
    ctx.strokeRect(pred_box_yx[1], pred_box_yx[0]-10, pred_box_yx2[1]-pred_box_yx[1], pred_box_yx2[0]-pred_box_yx[0]+10);// x, y , w, h
    document.getElementById("ans").innerHTML = classes[pred_class[ans_bbox_index[0]]];
  }catch(e){
    console.log('nothing')
  }
  file_or_realtime = 0
}


//--------------realtime
async function show(){
  if (file_or_realtime == 1){
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
  }
}


videoSelect.onchange = start;

start();

setInterval(show, 1000 / 999);
