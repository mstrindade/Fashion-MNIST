/* new function */
function load_variables(){
  $("#caminhos_variables").html("<center><img src='https://flevix.com/wp-content/uploads/2019/07/Facebook-Loading-Icons-1.gif' style='height: 300px; margin-top: 3cm;;'></center>");
  $.ajax({
      url: "/caminhos_variables",
      type: "GET",
      contentType: 'application/json;charset=UTF-8',
      data: {
          'variable': document.getElementById('model_variable').value,
          'sub': document.getElementById('os_number').value,
          'ord': document.getElementById('choose_feature').value

      },
      dataType:"json",
      success: function (data) {
          $("#caminhos_variables").html("");
          Plotly.newPlot('caminhos_variables', data);
      }
  });
}

/* new function */
function load_map(){
  $("#caminhos_map").html("<center><img src='https://flevix.com/wp-content/uploads/2019/07/Facebook-Loading-Icons-1.gif' style='height: 300px; margin-top: 3cm;;'></center>");
  $.ajax({
      url: "/caminhos_map",
      type: "GET",
      contentType: 'application/json;charset=UTF-8',
      data: {
          'variable': document.getElementById('model_variable').value,
          'sub': document.getElementById('os_number').value,
          'ord': document.getElementById('choose_feature').value

      },
      dataType:"json",
      success: function (data) {
          $("#caminhos_map").html("");
          Plotly.newPlot('caminhos_map', data);
      }
  });
}

$('#os_number').on('change',function(){
  load_variables();
  load_map();
});

$('#choose_feature').on('change',function(){
  load_variables();
  load_map();
});

$('#model_variable').on('change',function(){
  load_variables();
  load_map();
});

load_variables();
load_map();