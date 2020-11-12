var params;


const request = async () => {
    const response = await fetch('../json/params.json');
    const json = await response.json();
    params = json;
}

request();

var age_field = document.getElementById('age');
var trestbps_field = document.getElementById('trestbps');
var chol_field = document.getElementById('chol');
var thalach_field = document.getElementById('thalach');
var oldpeak_field = document.getElementById('oldpeak');

var sex_field = document.getElementById('sex');
var cp_field = document.getElementById('cp');
var fbs_field = document.getElementById('fbs');
var restecg_field = document.getElementById('restecg');
var exang_field = document.getElementById('exang');
var slope_field = document.getElementById('slope');
var ca_field = document.getElementById('ca');
var thal_field = document.getElementById('thal');


document.getElementById('submit').onclick = function(){
    var features = [];

    var num_fields = [age_field, trestbps_field, chol_field, thalach_field, oldpeak_field];
    var num_values = [];

    for (i=0; i<num_fields.length; i++) {
        var value = (num_fields[i].value - params.ss_mean[i]) / params.ss_std[i];
        num_values.push(value);
    }


    var cat_fields = [sex_field, cp_field, fbs_field, restecg_field, exang_field, slope_field, ca_field, thal_field];
    var cat_values = [];

    for (i=0; i<cat_fields.length; i++) {
        var field_arr = new Array(cat_fields[i].length - 1).fill(0);
        field_ind = cat_fields[i].selectedIndex - 1;
        field_arr[field_ind] = 1;
        cat_values.push(...field_arr);
    }
   
    features = num_values.concat(cat_values);

    var prediction = clf.predict(features);
    console.log(prediction);

}