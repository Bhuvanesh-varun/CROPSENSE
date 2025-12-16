
// GeoJSON for Indian States boundaries
const GEOJSON_URL =
  "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson";
const DATA_FILE = "/static/data.json";

let geojsonData = null;
let rawCropData = null;

const svg = d3.select("#map");
const width = svg.node().getBoundingClientRect().width;
const height = svg.node().getBoundingClientRect().height;
const projection = d3
  .geoMercator()
  .fitSize([width, height], { type: "FeatureCollection", features: [] });
const path = d3.geoPath().projection(projection);
const tooltip = d3.select("body").append("div").attr("class", "tooltip");
const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

const legendWidth = 300;
const legendHeight = 20;
const legendSvg = d3
  .select("#map-legend")
  .append("svg")
  .attr("width", legendWidth)
  .attr("height", legendHeight + 25);

// --- Core Logic Functions ---
async function init() {
  try {
    const [geoResponse, dataResponse] = await Promise.all([
      fetch(GEOJSON_URL),
      fetch(DATA_FILE),
    ]);

    geojsonData = await geoResponse.json();
    rawCropData = await dataResponse.json();

    if (!rawCropData || rawCropData.length === 0) {
      console.error("Failed to load or data file is empty.");
      return;
    }

    const uniqueCrops = [...new Set(rawCropData.map((d) => d.cropname))].sort();
    const select = d3.select("#crop-select");

    select
      .selectAll("option")
      .data(uniqueCrops)
      .enter()
      .append("option")
      .attr("value", (d) => d)
      .text((d) => d);

    const mapElement = d3.select("#map").node();
    const mapWidth = mapElement.getBoundingClientRect().width;
    const mapHeight = mapElement.getBoundingClientRect().height;

    projection.fitSize([mapWidth, mapHeight], geojsonData);

    const initialCrop = uniqueCrops[0] || "";
    updateMap(initialCrop);

    select.on("change", function () {
      updateMap(this.value);
    });
  } catch (error) {
    console.error(
      "Error loading required files. Ensure 'data.json' is in the correct path and GeoJSON URL is accessible.",
      error
    );
  }
}

function updateMap(selectedCrop) {
  const currentCropData = rawCropData.filter(
    (d) => d.cropname === selectedCrop
  );
  const productionMap = new Map();
  let maxProduction = 0;

  currentCropData.forEach((d) => {
    const production = parseFloat(d["Production (Lakh Tonnes)"]) || 0;
    if (d.statename !== "Others") {
      productionMap.set(d.statename, production);
      if (production > maxProduction) {
        maxProduction = production;
      }
    }
  });

  const states = svg
    .selectAll(".state")
    .data(geojsonData.features, (d) => d.properties.ST_NM);

  states
    .enter()
    .append("path")
    .attr("class", "state")
    .attr("d", path)
    .merge(states)
    .on("mouseover", handleMouseOver)
    .on("mousemove", handleMouseMove)
    .on("mouseout", handleMouseOut);
  states.exit().remove();
  updateTopProducers(selectedCrop, currentCropData);
}

function updateTopProducers(selectedCrop, currentCropData) {
  d3.select("#current-crop").text(selectedCrop);

  // --- Group by state and sum production ---
  const stateProductionMap = new Map();

  currentCropData.forEach(d => {
    if (d.statename !== "Others") {
      const production = parseFloat(d["Production (Lakh Tonnes)"]) || 0;

      if (!stateProductionMap.has(d.statename)) {
        stateProductionMap.set(d.statename, production);
      } else {
        stateProductionMap.set(
          d.statename,
          stateProductionMap.get(d.statename) + production
        );
      }
    }
  });

  // Convert map to array
  const aggregatedData = Array.from(stateProductionMap, ([statename, Production]) => ({
    statename,
    Production_Lakh_Tonnes: Production,
  }));

  // Sort and take top 5
  const top5 = aggregatedData
    .sort((a, b) => b.Production_Lakh_Tonnes - a.Production_Lakh_Tonnes)
    .slice(0, 5);

  const list = d3.select("#top-list").html("");

  if (top5.length === 0 || top5[0].Production_Lakh_Tonnes === 0) {
    list.append("li").text("No significant production data available for this crop.");
    return;
  }

  top5.forEach((item, index) => {
    const formattedProduction = item.Production_Lakh_Tonnes.toFixed(2);
    list.append("li").text(
      `${index + 1}. ${item.statename}: ${formattedProduction} Lakh Tonnes`
    );
  });
}

function handleMouseOver(event, d) {
  const stateName = d.properties.ST_NM;
  d3.select(this).attr("stroke", "#000").attr("stroke-width", 1.5);

  const selectedCrop = d3.select("#crop-select").property("value");
  const stateData = rawCropData.find(
    (item) => item.cropname === selectedCrop && item.statename === stateName
  );

  const production = stateData
    ? parseFloat(stateData["Production (Lakh Tonnes)"])
        .toFixed(2)
        .toLocaleString()
    : "N/A";

  tooltip.style("opacity", 1).html(`
            <strong>${stateName}</strong><br>
            Production: ${production} Lakh Tonnes
        `);
}

function handleMouseMove(event) {
  tooltip
    .style("left", event.pageX + 10 + "px")
    .style("top", event.pageY - 28 + "px");
}

function handleMouseOut(event, d) {
  d3.select(this).attr("stroke", "#333").attr("stroke-width", 0.5);
  tooltip.style("opacity", 0);
}
init();

window.addEventListener("resize", function () {
    if (window.myMap) {
        setTimeout(() => {
            window.myMap.invalidateSize();
        }, 300);
    }
});

setTimeout(() => {
    if (window.myMap) {
        window.myMap.invalidateSize();
    }
}, 500);

window.addEventListener("resize", () => {
    if (window.myMap) {
        setTimeout(() => window.myMap.invalidateSize(), 300);
    }
});

setTimeout(() => {
    if (window.myMap) {
        window.myMap.invalidateSize();
    }
}, 500);
