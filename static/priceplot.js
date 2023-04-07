 $(document).ready(function(){
        $(".updateButton").on("click", function(){
            var chartId = $(this).attr("chartId")
            var chartOption = $("#chartOption").val()
            console.log(chartId)
            console.log(chartOption)

            req = $.ajax({
                url : "/updatepriceplot",
                type : "POST",
                data : {chartId: chartId, chartOption : chartOption}
            });

            req.done(function(data) {
                $("#chartSectionchart1").fadeOut(100).fadeIn(100);
                var graphs = JSON.parse(data);
                console.log(graphs)
                Plotly.react('graph', graphs, {});
            });
        });
    });

