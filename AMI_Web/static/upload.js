function showLabel(filename, damage) {
    $.ajax({
        type: "POST",
        url: "/showLabels",
        data: {"damage" : damage, "filename" : filename},
        success: function() {
            document.getElementById(filename+"_label").innerHTML = 
                "<div class='u-container-layout u-container-layout-4 u-label-exist'>"
                    + "<h4 class='u-text u-text-default u-grid-label'>" + damage + "</h4>"
                "</div>"
        }
    })
}

function checkComplete() {
    $.ajax({
        type: "POST",
        url: "/checkComplete",
        success: function (response) {
            if (response.complete=="True") {
                $.ajax({
                    type: "POST",
                    url: "/exportLabels",
                    success: function (response) {
                        download("label.json", response)
                    }
                })
            } else {
                if (confirm("Not all images are labeled. Do you want to continue?")) {
                    $.ajax({
                        type: "POST",
                        url: "/exportLabels",
                        success: function (response) {
                            download("label.json", response)
                        }
                    })
                }
            }
        },
    });
}

function download(filename, textInput) {
    var element = document.createElement('a');
    element.setAttribute('href','data:text/plain;charset=utf-8,' + encodeURIComponent(textInput));
    element.setAttribute('download', filename);
    document.body.appendChild(element);
    element.click();
}

function exitUploadAlert(hyperlink) {
    $.ajax({
        type: "POST",
        url: "/checkExportDone",
        success: function (response) {
            if (response.done == "True") {
                $.ajax({
                    type: "POST",
                    url: "/clearDisplayed",
                    success: function () {
                        window.location = hyperlink
                    }
                })
            } else {
                if (confirm("Your data will be lost. Do you wish to continue?")) {
                    $.ajax({
                        type: "POST",
                        url: "/clearDisplayed",
                        success: function () {
                            window.location = hyperlink
                        }
                    })
                }
            }
        }
    })
    
}

function showCorrections(filename, damage) {
    $.ajax({
        type: "POST",
        url: "/showCorrections",
        data: {"damage" : damage, "filename" : filename},
        success: function(response) {
            if (response.changed == "True") {
                document.getElementById(filename+"_predict").innerHTML = 
                    "<div class='u-container-style u-group u-group-3'>"
                        + "<div class='u-container-layout u-corrected'>"
                            + "<h5 class='u-text u-text-default u-grid-label'>Correction: " + damage + "</h5>"
                        + "</div>"
                    + "</div>"
            } else {
                document.getElementById(filename+"_predict").innerHTML = ""
            }
        }
    })
}

function exitPredictAlert(hyperlink) {
    if (confirm("Your data will be lost. Do you wish to continue?")) {
        $.ajax({
            type: "POST",
            url: "/clearDisplayed",
            success: function () {
                window.location = hyperlink
            }
        })
    }
}

function correctionsDone() {
    $.ajax({
        type: "POST",
        url: "/correctionsDone",
        success: function () {
            $.ajax({
                type: "POST",
                url: "/clearDisplayed",
                success: function () {
                    window.location = "/predictcorrect"
                }
            })
        },
        error: function () {
            alert("There have been no new corrections.")
        }
    })
}

function showButtonsUpload(filename, image_count) {
    document.getElementById(filename+"_image").innerHTML = 
        "<div class='u-container-layout u-container-layout-14' onmouseleave='showImageUpload(" + JSON.stringify(filename) + ", " + JSON.stringify(image_count) + ")'>"
            + "<button type=submit class='u-border-none u-btn u-btn-round u-button-style u-btn-1'"
                + "onclick='showLabel(" + JSON.stringify(filename) + ", " + JSON.stringify("scratch") + ")'>scratch</button>"
            + "<button type=submit class='u-border-none u-btn u-btn-round u-button-style u-btn-1'"
                + "onclick='showLabel(" + JSON.stringify(filename) + ", " + JSON.stringify("dent") + ")'>dent</button>"
            + "<button type=submit class='u-border-none u-btn u-btn-round u-button-style u-btn-1'"
                + "onclick='showLabel(" + JSON.stringify(filename) + ", " + JSON.stringify("rim") + ")'>rim</button>"
            + "<button type=submit class='u-border-none u-btn u-btn-round u-button-style u-btn-1'"
                + "onclick='showLabel(" + JSON.stringify(filename) + ", " + JSON.stringify("other") + ")'>other</button>"
        + "</div>"
}

function showImageUpload(filename, image_count) {
    $.ajax({
        type: "POST",
        url: "/currentImageUpload",
        data: {"image_count" : image_count},
        success: function (response) {
            document.getElementById(filename+"_image").innerHTML =
                "<div class='u-container-layout u-container-layout-2' onmouseover='showButtonsUpload(" + JSON.stringify(filename) + ", " + JSON.stringify(image_count) + ")'>"
                    + "<img class='u-image u-image-round u-radius-10 u-grid-image' src=" + JSON.stringify(response) + "alt='Image not available'>"
                + "</div>"
        }
    })
}

function showButtonsPredict(filename, image_count) {
    document.getElementById(filename+"_image").innerHTML = 
        "<div class='u-container-layout u-container-layout-14' onmouseleave='showImagePredict(" + JSON.stringify(filename) + ", " + JSON.stringify(image_count) + ")'>"
            + "<button type=submit class='u-border-none u-btn u-btn-round u-button-style u-btn-1'"
                + "onclick='showCorrections(" + JSON.stringify(filename) + ", " + JSON.stringify("scratch") + ")'>scratch</button>"
            + "<button type=submit class='u-border-none u-btn u-btn-round u-button-style u-btn-1'"
                + "onclick='showCorrections(" + JSON.stringify(filename) + ", " + JSON.stringify("dent") + ")'>dent</button>"
            + "<button type=submit class='u-border-none u-btn u-btn-round u-button-style u-btn-1'"
                + "onclick='showCorrections(" + JSON.stringify(filename) + ", " + JSON.stringify("rim") + ")'>rim</button>"
            + "<button type=submit class='u-border-none u-btn u-btn-round u-button-style u-btn-1'"
                + "onclick='showCorrections(" + JSON.stringify(filename) + ", " + JSON.stringify("other") + ")'>other</button>"
        + "</div>"
}

function showImagePredict(filename, image_count) {
    $.ajax({
        type: "POST",
        url: "/currentImagePredict",
        data: {"image_count" : image_count},
        success: function (response) {
            document.getElementById(filename+"_image").innerHTML =
                "<div class='u-container-layout u-container-layout-2' onmouseover='showButtonsPredict(" + JSON.stringify(filename) + ", " + JSON.stringify(image_count) + ")'>"
                    + "<img class='u-image u-image-round u-radius-10 u-grid-image' src=" + JSON.stringify(response) + "alt='Image not available'>"
                + "</div>"
        }
    })
}
