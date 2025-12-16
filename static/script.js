document.addEventListener("DOMContentLoaded", function () {

  AOS?.init({ duration: 700, once: true });

  /* ======================================================
     DOM REFERENCES
  ====================================================== */
  const aiOverlay = document.getElementById("aiFormOverlay");
  const aiMainContent = document.getElementById("aiMainContent");
  const closeFormBtn = document.getElementById("closeFormBtn");
  const prevBtn = document.getElementById("prevStepBtn");
  const getRecoBtn = document.getElementById("getRecommendationBtn");

  const steps = Array.from(document.querySelectorAll(".step-card"));
  const progressBar = document.getElementById("progressBar");
  const progressLabel = document.getElementById("curStepNum");

  const resultPage = document.getElementById("recommendationResultsArea");
  const planHtmlContainer = document.getElementById("planHtmlContainer");
  const inputSummary = document.getElementById("inputSummary");

  const leafPercentBox = document.getElementById("leafPercentBox");

  let currentStep = 0;
  const totalSteps = steps.length;
  const stepProgress = 100 / totalSteps;

/* ======================================================
   RESTORE LAST SESSION (recommendations.html only)
====================================================== */

  async function loadLastRecommendation() {
    try {
      const res = await fetch("/last_recommendation", {
        credentials: "include"
      });

      const json = await res.json();
      if (json.status !== "success") return;

      const data = json.data;

      // 🔥 stop loading animation
      document.getElementById("loadingState")?.classList.add("d-none");
      document.getElementById("recommendationResultsArea")?.classList.remove("d-none");

      // -----------------------------
      // Populate Summary of Inputs
      // -----------------------------
      const inputs = data.inputs || {};
      for (const [key, val] of Object.entries(inputs)) {
        const el = document.querySelector(`[data-input-key="${key}"]`);
        if (el) el.innerText = val ?? "—";
      }

      // -----------------------------
      // Populate SQI / PHI
      // -----------------------------
      document.getElementById("sqiValue").innerText = data.sqi;
      document.getElementById("phiValue").innerText = data.phi;
      document.getElementById("sqiClass").innerText = data.sqi_class;
      document.getElementById("phiClass").innerText = data.phi_class;

      // -----------------------------
      // Treatments / Plan
      // -----------------------------
      if (data.plan_text) {
        document.getElementById("treatmentText").innerText = data.plan_text;
      }

      if (data.plan_html) {
        document.getElementById("treatmentHtml").innerHTML = data.plan_html;
      }

    } catch (e) {
      console.error("Replay load failed:", e);
    }
  }

  // 🔥 AUTO LOAD HISTORY ON PAGE OPEN
  loadLastRecommendation();

/* ======================================================
   RESTORE LAST SESSION (recommendations.html only)
====================================================== */
  fetch("/last_recommendation")
    .then(res => res.json())
    .then(res => {
      if (res.status !== "success") return;

      const payload = res.data;

      // Hide hero + form
      aiMainContent.classList.add("d-none");
      aiOverlay.classList.add("d-none");

      // Show results
      resultPage.classList.remove("d-none");

      // KPI
      document.getElementById("kpiSection").classList.remove("d-none");
      document.getElementById("sqiValue").textContent = payload.sqi;
      document.getElementById("phiValue").textContent = payload.phi;
      document.getElementById("sqiClass").textContent = payload.sqi_class;
      document.getElementById("phiClass").textContent = payload.phi_class;

      // Summary + treatments
      renderSummaryPremium(payload.inputs);
      planHtmlContainer.classList.remove("d-none");
      planHtmlContainer.innerHTML = renderTreatments(payload.treatments);
    });

  /* ======================================================
     INPUT STATE
  ====================================================== */
  const inputs = {
    crop: null,
    previousCrop: null,
    soilType: null,
    soilTexture: null,
    growthStage: null,
    irrigationType: null,
    irrigationStatus: null,
    irrigationCount: 0,
    irrigationCountTouched: false,
    leafColor: null,
    spots: null,
    pests: null,
    leafYellowPercent: 0,
    leafYellowTouched: false,
    usedFertilizer: "No",
    fertilizerType: "",
    fertilizerQty: 0,
    usedPesticide: "No",
    pesticideType: "",
    pesticideQty: 0,
    usedFungicide: "No",
    fungSprays: 0
  };

  /* ======================================================
     INPUT SUMMARY
  ====================================================== */

  function renderSummaryPremium(inputs) {
    let html = `<div class="row g-2">`;

    const labels = {
      crop: "Crop",
      previousCrop: "Previous Crop",
      soilType: "Soil Type",
      soilTexture: "Soil Texture",
      growthStage: "Growth Stage",
      irrigationType: "Irrigation Type",
      irrigationStatus: "Soil Moisture",
      irrigationCount: "Irrigation Count",
      leafColor: "Leaf Color",
      leafYellowPercent: "Leaf Yellowing (%)",
      spots: "Leaf Spots",
      pests: "Pest Incidence"
    };

    Object.entries(labels).forEach(([key, label]) => {
      const value = inputs[key] ?? "—";
      html += `
      <div class="col-6">
        <div class="input-pill">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      </div>
    `;
    });

    html += `</div>`;
    inputSummary.innerHTML = html;
  }
  function normalizeBackendInputs(raw) {
    return {
      crop: raw.Crop_Name,
      previousCrop: raw.Previous_Crop,
      soilType: raw.Soil_Type,
      soilTexture: raw.Soil_Texture_Class,
      growthStage: raw.Growth_Stage,
      irrigationType: raw.Irrigation_Type,
      irrigationStatus: raw.Current_Soil_State,
      irrigationCount: raw.No_Of_Irrigations_Since_Sowing,
      leafColor: raw.Leaf_Colour,
      leafYellowPercent: raw.Leaf_Yellowing_Percent,
      spots: raw.Leaf_Spot_Severity,
      pests: raw.Pest_Incidence
    };
  }

  function renderTreatments(treatments) {
    if (!treatments || !treatments.length) {
      return `<div class="alert alert-success">🌱 No treatment required. Crop is healthy.</div>`;
    }

    return treatments.map((t, i) => `
  <div class="treatment-card">
    <div class="treatment-index">${i + 1}</div>
    <div>
      <h6>${t.Issue}</h6>
      <p><strong>Fertilizer:</strong> ${t.Fertilizer}</p>
      <p><strong>Dose:</strong> ${t.Dose}</p>
      <small>${t.Notes || ""}</small>
    </div>
  </div>
`).join("");
  }


  /* ======================================================
     VALIDATION (UPDATED FOR 7 STEPS)
  ====================================================== */
  function isStepComplete(i) {
    if (i === 0) return inputs.crop;
    if (i === 1) return inputs.previousCrop;
    if (i === 2) return inputs.soilType && inputs.soilTexture;
    if (i === 3) return inputs.growthStage;
    if (i === 4)
      return inputs.irrigationType &&
        inputs.irrigationStatus &&
        inputs.irrigationCountTouched;
    if (i === 5)
      return inputs.leafColor &&
        inputs.spots !== null &&
        inputs.pests !== null &&
        inputs.leafYellowTouched;
    if (i === 6) return true; // final step
    return false;
  }

  function autoNextStep() {
    if (isStepComplete(currentStep) && currentStep < totalSteps - 1) {
      showStep(currentStep + 1);
    }
  }

  /* ======================================================
     STEP CONTROL
  ====================================================== */
  function showStep(n) {
    if (n < 0 || n >= totalSteps) return;

    steps.forEach((s, i) => s.classList.toggle("active", i === n));
    currentStep = n;

    progressBar.style.width = `${stepProgress * (n + 1)}%`;
    progressLabel.textContent = n + 1;

    prevBtn.style.display = n === 0 ? "none" : "inline-block";
    getRecoBtn.classList.toggle("d-none", n !== totalSteps - 1);

    aiOverlay.scrollTop = 0;
  }

  /* ======================================================
     IMAGE SELECTIONS
  ====================================================== */
  document.querySelectorAll(".img-selection").forEach(box => {
    box.addEventListener("click", () => {
      const field = box.dataset.field;
      const value = box.dataset.value;
      inputs[field] = value;

      box.closest(".selection-grid")
        ?.querySelectorAll(".img-selection")
        .forEach(el => el.classList.remove("selected"));

      box.classList.add("selected");

      renderSummaryPremium(inputs);
      autoNextStep();
    });
  });

  /* ======================================================
     RADIO INPUTS
  ====================================================== */

  document.querySelectorAll("input[type=radio]").forEach(radio => {
    radio.addEventListener("change", e => {
      inputs[e.target.name] = e.target.value;

      // FERTILIZER
      if (e.target.name === "usedFertilizer" && e.target.value === "No") {
        inputs.fertilizerType = "none";
        inputs.fertilizerQty = 0;
      }

      // PESTICIDE
      if (e.target.name === "usedPesticide" && e.target.value === "No") {
        inputs.pesticideType = "none";
        inputs.pesticideQty = 0;
      }

      // FUNGICIDE
      if (e.target.name === "usedFungicide" && e.target.value === "No") {
        inputs.fungSprays = 0;
      }

      document.getElementById("fertilizerDetails")
        ?.classList.toggle("d-none", inputs.usedFertilizer === "No");
      document.getElementById("pesticideDetails")
        ?.classList.toggle("d-none", inputs.usedPesticide === "No");
      document.getElementById("fungicideDetails")
        ?.classList.toggle("d-none", inputs.usedFungicide === "No");

      renderSummaryPremium(inputs);
      autoNextStep();
    });
  });


  /* ======================================================
     SLIDERS & NUMBERS (FIXED)
  ====================================================== */
  document.getElementById("leafYellowRange")?.addEventListener("input", e => {
    inputs.leafYellowTouched = true;
    inputs.leafYellowPercent = Number(e.target.value);
    leafPercentBox.textContent = `${e.target.value}%`;
    renderSummaryPremium(inputs);
  });

  document.getElementById("irrigationCount")?.addEventListener("input", e => {
    inputs.irrigationCountTouched = true;
    inputs.irrigationCount = Number(e.target.value);
    renderSummaryPremium(inputs);
  });
  // Fertilizer type
  document.getElementById("fertilizerType")?.addEventListener("change", e => {
    inputs.fertilizerType = e.target.value || "none";
    renderSummaryPremium(inputs);
  });
  document.getElementById("fertQtyRange")?.addEventListener("input", e => {
    inputs.fertilizerQty = Number(e.target.value);
    document.getElementById("fertQtyBox").textContent = e.target.value;
  });
  // Pesticide type
  document.getElementById("pesticideType")?.addEventListener("change", e => {
    inputs.pesticideType = e.target.value || "none";
    renderSummaryPremium(inputs);
  });
  document.getElementById("pestQtyRange")?.addEventListener("input", e => {
    inputs.pesticideQty = Number(e.target.value);
    document.getElementById("pestQtyBox").textContent = e.target.value;
  });
  document.getElementById("fungSprays")?.addEventListener("input", e => {
    inputs.fungSprays = Number(e.target.value);
  });



  /* ======================================================
     PREVIOUS BUTTON
  ====================================================== */
  prevBtn.addEventListener("click", () => {
    if (currentStep > 0) showStep(currentStep - 1);
  });

  /* ======================================================
     CLOSE FORM
  ====================================================== */
  closeFormBtn.addEventListener("click", () => {
    aiOverlay.classList.add("d-none");
    aiMainContent.classList.remove("d-none");
    currentStep = 0;
    showStep(0);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  /* ======================================================
     GET RECOMMENDATION (FIXED)
  ====================================================== */
  getRecoBtn.addEventListener("click", async () => {

    aiOverlay.classList.add("d-none");
    resultPage.classList.remove("d-none");

    aiLoadingState.classList.remove("d-none");
    planHtmlContainer.classList.add("d-none");

    const res = await fetch("/get_recommendation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputs)
    });

    const contentType = res.headers.get("content-type") || "";

    if (res.status === 401) {
      new bootstrap.Modal(
        document.getElementById("loginRequiredModal")
      ).show();
      return;
    }

    if (!res.ok || !contentType.includes("application/json")) {
      const text = await res.text();
      console.error("Server returned non-JSON:", text);
      alert("Unexpected server error. Please try again.");
      return;
    }

    const data = await res.json();
    // --------------------------------------------
    // STEP 3.1 — HANDLE QUOTA LIMIT REACHED
    // --------------------------------------------
    if (data.status === "limit_reached") {
      alert("Your free trials are exhausted. Please upgrade your plan.");
      window.location.href = "/plans";
      return;
    }

    const payload = data.data;

    // KPI
    document.getElementById("kpiSection").classList.remove("d-none");
    document.getElementById("kpiSection").style.display = "flex";

    document.getElementById("sqiValue").textContent = payload.sqi;
    document.getElementById("phiValue").textContent = payload.phi;
    document.getElementById("sqiClass").textContent = payload.sqi_class;
    document.getElementById("phiClass").textContent = payload.phi_class;

    // Inputs
    const normalizedInputs = normalizeBackendInputs(payload.inputs);
    renderSummaryPremium(normalizedInputs);


    // Treatments
    aiLoadingState.classList.add("d-none");
    planHtmlContainer.classList.remove("d-none");
    planHtmlContainer.innerHTML = renderTreatments(payload.treatments);
  });





  /* ======================================================
   START NEW ANALYSIS
====================================================== */
  const startNewBtn = document.getElementById("startNewAnalysisBtn");

  startNewBtn?.addEventListener("click", () => {

    // Hide results
    resultPage.classList.add("d-none");

    // Reset inputs
    Object.keys(inputs).forEach(k => {
      if (typeof inputs[k] === "number") inputs[k] = 0;
      else inputs[k] = null;
    });
    inputs.usedFertilizer = "No";
    inputs.usedPesticide = "No";
    inputs.usedFungicide = "No";

    // Reset UI
    renderSummaryPremium(inputs);

    // Reset steps
    currentStep = 0;
    showStep(0);

    // Show form again
    aiOverlay.classList.remove("d-none");
    aiOverlay.scrollIntoView({ behavior: "smooth" });
  });

  /* ======================================================
     download receipt button
  ====================================================== */

  document.getElementById("downloadReportBtn")?.addEventListener("click", async () => {
    const resultArea = document.getElementById("recommendationResultsArea");
    if (!resultArea) return;

    const canvas = await html2canvas(resultArea, { scale: 2 });
    const imgData = canvas.toDataURL("image/png");

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF("p", "mm", "a4");

    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

    pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
    pdf.save("CropSense_AI_Report.pdf");
  });




  /* ======================================================
     INIT
  ====================================================== */
  showStep(0);
  renderSummaryPremium(inputs);
});