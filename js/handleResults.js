result_heading = document.getElementById("result");

var query = decodeURIComponent(window.location.search).split("=")[1];

var decoded_query = atob(query);

var [prediction, time] = decoded_query.split('|');

var time_delta = Date.now() - time;

if (time_delta > 5000) window.location.href = "index.html";

if (prediction == "1") {
    result_heading.innerText = 'You have Heart Disease';
} else {
    result_heading.innerText = 'You Don\'t have Heart Disease';
}