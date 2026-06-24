const MODEL_URL = "https://cdn.glitch.me/2046b88b-673a-457f-b1b8-7169ce9bf13a/deoldify-quant.onnx";
const MAX_FILE_SIZE = 5 * 1024 * 1024;
const MODEL_SIZE = 256;

const imageInput = document.getElementById("imageInput");
const originalCanvas = document.getElementById("originalCanvas");
const resultCanvas = document.getElementById("resultCanvas");
const statusEl = document.getElementById("status");
const downloadBtn = document.getElementById("downloadBtn");
const overlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");

let session = null;
let lastResultUrl = null;

const setStatus = (message, isError = false) => {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#fca5a5" : "#cbd5e1";
};

const showOverlay = (message) => {
  loadingText.textContent = message;
  overlay.style.display = "flex";
};

const hideOverlay = () => {
  overlay.style.display = "none";
};

const loadModel = async () => {
  if (session) {
    return session;
  }

  showOverlay("กำลังโหลดโมเดล AI...");
  try {
    session = await ort.InferenceSession.create(MODEL_URL);
    setStatus("โมเดลพร้อมใช้งานแล้ว. อัปโหลดภาพเพื่อเติมสีได้เลย.");
    return session;
  } catch (error) {
    setStatus("ไม่สามารถโหลดโมเดลได้ โปรดลองรีเฟรชหรือเชื่อมต่ออินเทอร์เน็ตอีกครั้ง.", true);
    console.error(error);
    throw error;
  } finally {
    hideOverlay();
  }
};

const isSupportedFile = (file) => {
  return file && (file.type === "image/png" || file.type === "image/jpeg");
};

const fileIsTooLarge = (file) => {
  return file && file.size > MAX_FILE_SIZE;
};

const imageDataToTensor = (imageData) => {
  const width = imageData.width;
  const height = imageData.height;
  const floatArr = new Float32Array(width * height * 3);
  const data = imageData.data;
  const pixelCount = width * height;

  for (let i = 0; i < pixelCount; i += 1) {
    const src = i * 4;
    floatArr[i] = data[src];
    floatArr[i + pixelCount] = data[src + 1];
    floatArr[i + pixelCount * 2] = data[src + 2];
  }

  return new ort.Tensor("float32", floatArr, [1, 3, height, width]);
};

const tensorToImageData = (tensor) => {
  const [batch, channel, height, width] = tensor.dims;
  const tensorData = tensor.data ? tensor.data : tensor.cpuData;
  const output = new ImageData(width, height);
  const pixels = output.data;
  const slice = height * width;

  for (let h = 0; h < height; h += 1) {
    for (let w = 0; w < width; w += 1) {
      const pixelIndex = h * width + w;
      const r = Math.round(tensorData[pixelIndex]);
      const g = Math.round(tensorData[pixelIndex + slice]);
      const b = Math.round(tensorData[pixelIndex + slice * 2]);
      const dst = pixelIndex * 4;

      pixels[dst] = Math.min(255, Math.max(0, r));
      pixels[dst + 1] = Math.min(255, Math.max(0, g));
      pixels[dst + 2] = Math.min(255, Math.max(0, b));
      pixels[dst + 3] = 255;
    }
  }

  return output;
};

const drawCanvas = (canvas, image, width, height) => {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0, width, height);
};

const processImage = async (file) => {
  if (!isSupportedFile(file)) {
    setStatus("กรุณาเลือกไฟล์ PNG หรือ JPG เท่านั้น.", true);
    return;
  }

  if (fileIsTooLarge(file)) {
    setStatus("ไฟล์ใหญ่เกินไป โปรดเลือกไฟล์ไม่เกิน 5MB.", true);
    return;
  }

  const image = new Image();
  const objectUrl = URL.createObjectURL(file);
  image.src = objectUrl;

  image.onload = async () => {
    URL.revokeObjectURL(objectUrl);
    drawCanvas(originalCanvas, image, image.width, image.height);
    downloadBtn.disabled = true;
    setStatus("กำลังประมวลผลภาพด้วย AI...");
    showOverlay("ประมวลผลภาพ... โปรดรอสักครู่");

    try {
      await loadModel();
      const bufferCanvas = document.createElement("canvas");
      bufferCanvas.width = MODEL_SIZE;
      bufferCanvas.height = MODEL_SIZE;
      const bufferCtx = bufferCanvas.getContext("2d");
      bufferCtx.drawImage(image, 0, 0, MODEL_SIZE, MODEL_SIZE);
      const inputImage = bufferCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
      const inputTensor = imageDataToTensor(inputImage);
      const feeds = { input: inputTensor };
      const results = await session.run(feeds);
      const outputTensor = results.out || results.output || Object.values(results)[0];
      const imageData = tensorToImageData(outputTensor);

      resultCanvas.width = image.width;
      resultCanvas.height = image.height;
      const resultCtx = resultCanvas.getContext("2d");
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = MODEL_SIZE;
      tempCanvas.height = MODEL_SIZE;
      tempCanvas.getContext("2d").putImageData(imageData, 0, 0);
      resultCtx.clearRect(0, 0, image.width, image.height);
      resultCtx.drawImage(tempCanvas, 0, 0, image.width, image.height);

      setStatus("เติมสีสำเร็จแล้ว! คลิกปุ่มดาวน์โหลดเพื่อบันทึกภาพ.");
      downloadBtn.disabled = false;
    } catch (error) {
      setStatus("เกิดข้อผิดพลาดขณะประมวลผลภาพ โปรดลองอีกครั้ง.", true);
      console.error(error);
    } finally {
      hideOverlay();
    }
  };

  image.onerror = () => {
    URL.revokeObjectURL(objectUrl);
    setStatus("ไม่สามารถโหลดรูปภาพได้ โปรดเลือกไฟล์ที่ถูกต้อง.", true);
  };
};

imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    processImage(file);
  }
});

downloadBtn.addEventListener("click", () => {
  const dataUrl = resultCanvas.toDataURL("image/png");
  const anchor = document.createElement("a");
  anchor.href = dataUrl;
  anchor.download = "recolorai-result.png";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
});

window.addEventListener("load", () => {
  setStatus("เตรียมพร้อมใช้ AI เติมสี. อัปโหลดภาพหรือรอให้โมเดลโหลด.");
  loadModel().catch(() => {
    setStatus("โหลดโมเดลล้มเหลว โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ต.", true);
  });
});
