$(document).ready(function() {
    $('#imageForm').submit(function(event) {
      event.preventDefault(); // Prevent form from refreshing the page
  
      // Create FormData object to send images
      var formData = new FormData(this);
  
      // Show loading message while processing
      $('#similarityPercentage').text('Processing...');
      $('#errorMessage').text('');
  
      // Send POST request to the backend
      $.ajax({
        url: '/compare',  // Make sure this matches the route in Flask
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
          // Show result
          $('#errorMessage').text('');
          $('#similarityPercentage').text(response.message);
        },
        error: function(xhr, status, error) {
          // Show error message
          var errorMessage = xhr.responseJSON ? xhr.responseJSON.error : 'An error occurred';
          $('#errorMessage').text(errorMessage);
          $('#similarityPercentage').text('0.00%');
        }
      });
    });
  
    // Preview the first image
    $('#image1').change(function() {
      var reader = new FileReader();
      reader.onload = function(e) {
        $('#image1Preview').html('<img src="' + e.target.result + '" class="img-fluid" />');
      };
      reader.readAsDataURL(this.files[0]);
    });
  
    // Preview the second image
    $('#image2').change(function() {
      var reader = new FileReader();
      reader.onload = function(e) {
        $('#image2Preview').html('<img src="' + e.target.result + '" class="img-fluid" />');
      };
      reader.readAsDataURL(this.files[0]);
    });
  });
  