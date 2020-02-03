$(function() {
    $('#uploadField').change(function () {
        $('#pseudoUploadButton').css('display', 'none');
        $('#spinner').css('display', 'block');
        $('#uploadForm').submit();
    });

    $('#predictBtn').click(function () {
        $('#pseudoUploadButton').attr('disabled', true);
        $('#uploadField').attr('disabled', true);

        $('#predictBtn').css('display', 'none');
        $('#predictingFile').css('display', 'block');
    });
});
