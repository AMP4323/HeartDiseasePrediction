result_heading = document.getElementById("result");
guidance_heading = document.getElementById("guidance");

var guidance = [];
var query = decodeURIComponent(window.location.search).split("=")[1];

var decoded_query = atob(query);

var [prediction, time, inputs_string] = decoded_query.split('|');

var time_delta = Date.now() - time;

if (time_delta > 5000) window.location.href = "index.html";

if (prediction == "1") {
    result_heading.innerText = 'You have Heart Disease';
    result_heading.style = 'color: white; background: red;'
    guidance.push('Get a Doctor\'s Appointment soon for thorough checkup');
} else {
    result_heading.innerText = 'You Don\'t have Heart Disease';
    result_heading.style = 'color: white; background: green;'
}
guidance.push("You can try following things at home: <br>");

var inputs = JSON.parse(inputs_string);

if (inputs['chol'] > 200) guidance.push("* Cholestrol is higher than 200 mg/dl, Reduce Saturated Fats to lower it.");

var target_thalach = 220 - inputs['age'];
if (inputs['thalach'] > target_thalach) guidance.push("* Exceeded Max Heart Rate, Reduce it by taking frequent breaks during work and exercise.")

if(inputs['oldpeak'] < 2.0) guidance.push("* ST Segment Depression is less than 2mm, It can lead to sudden Cardiac Arrest. Ask for Medical Help.")

for (var guide in guidance) {
    guidance_heading.innerHTML += guidance[guide] + "<br>";
}