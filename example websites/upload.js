Dropzone.options.myDropzone = {
    paramName: "files", // The name that will be used to transfer the file
    maxFilesize: 10, // MB
    parallelUploads: 1, //limits number of files processed to reduce server load
    acceptedFiles: 'image/*',
    init: function() {
        this.on("sending", function(file, xhr, formData) {
            // Will send the filename along with the file as POST data.
            formData.append("filename", file.name);
        });

        this.on("success", function(file, response) {
            // Check if the response contains compressed data
            if (response.zip) {
                // Decompress the 'Predictions' property
                var decompressed = pako.inflate(response.Predictions, { to: 'string' });

                // Parse the decompressed data
                response.Predictions = JSON.parse(decompressed);
            }

            // Process the data
            response.Predictions.forEach(function(item) {
                // Create a new row
                var row = $('<tr></tr>');

                // Add data to the row
                row.append('<td>' + item['filename'] + '</td>');
                row.append('<td>' + item['predicted group'] + '</td>');
                row.append('<td>' + item['distance'] + '</td>');

                // Add the row to the table
                $('#responseTable').append(row);
            });
        })
        
    }
};
