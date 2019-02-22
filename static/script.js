$(function(){
    $("form").submit(function(){
        $("#submit-btn").prop("disabled", true);
        return true;
    });
});
