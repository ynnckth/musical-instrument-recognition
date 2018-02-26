$(function() {
    var myBlob = new Blob(["This is my blob content"], {type : "text/plain"});

    $('#uploadField').change(function () {
        $('#pseudoUploadButton').css('display', 'none');
        $('#loadingFile').css('display', 'block');
        $('#uploadForm').submit();
    });

    $('#predictBtn').click(function () {
        $('#pseudoUploadButton').attr('disabled', true);
        $('#uploadField').attr('disabled', true);

        $('#predictBtn').css('display', 'none');
        $('#predictingFile').css('display', 'block');
    });

    $('#bla').click(function () {
        $('#uploadForm').append("file", myBlob, "myfile.txt");
        $('#uploadForm').submit();
    });
});
