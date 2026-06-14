<?php
  /**
  * Requires the "PHP Email Form" library
  * The "PHP Email Form" library is available only in the pro version of the template
  * The library should be uploaded to: vendor/php-email-form/php-email-form.php
  * For more info and help: https://bootstrapmade.com/php-email-form/
  */

  // Only accept POST requests
  if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    die('Method Not Allowed');
  }

  // Validate email is present and non-empty
  if (!isset($_POST['email']) || trim($_POST['email']) === '') {
    http_response_code(400);
    die('Missing required field: email');
  }

  // Validate email format
  $email = filter_var(trim($_POST['email']), FILTER_VALIDATE_EMAIL);
  if ($email === false) {
    http_response_code(400);
    die('Invalid email address');
  }

  // Replace contact@example.com with your real receiving email address
  $receiving_email_address = 'contact@example.com';

  if( file_exists($php_email_form = '../assets/vendor/php-email-form/php-email-form.php' )) {
    include( $php_email_form );
  } else {
    die( 'Unable to load the "PHP Email Form" Library!');
  }

  $contact = new PHP_Email_Form;
  $contact->ajax = true;
  
  $contact->to = $receiving_email_address;
  $contact->from_name = $email;
  $contact->from_email = $email;
  $contact->subject = "New Subscription: " . $email;

  // Uncomment below code if you want to use SMTP to send emails. You need to enter your correct SMTP credentials
  /*
  $contact->smtp = array(
    'host' => 'example.com',
    'username' => 'example',
    'password' => 'pass',
    'port' => '587'
  );
  */

  $contact->add_message( $email, 'Email');

  $result = $contact->send();
  if ($result !== 'OK') {
    http_response_code(500);
  }
  echo $result;
?>
