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

  // Validate required fields are present and non-empty
  $required_fields = ['name', 'email', 'subject', 'message'];
  foreach ($required_fields as $field) {
    if (!isset($_POST[$field]) || trim($_POST[$field]) === '') {
      http_response_code(400);
      die("Missing required field: $field");
    }
  }

  // Validate email format
  $email = filter_var(trim($_POST['email']), FILTER_VALIDATE_EMAIL);
  if ($email === false) {
    http_response_code(400);
    die('Invalid email address');
  }

  // Sanitize inputs
  $name = htmlspecialchars(trim($_POST['name']), ENT_QUOTES, 'UTF-8');
  $subject = htmlspecialchars(trim($_POST['subject']), ENT_QUOTES, 'UTF-8');
  $message = htmlspecialchars(trim($_POST['message']), ENT_QUOTES, 'UTF-8');

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
  $contact->from_name = $name;
  $contact->from_email = $email;
  $contact->subject = $subject;

  // Uncomment below code if you want to use SMTP to send emails. You need to enter your correct SMTP credentials
  /*
  $contact->smtp = array(
    'host' => 'example.com',
    'username' => 'example',
    'password' => 'pass',
    'port' => '587'
  );
  */

  $contact->add_message( $name, 'From');
  $contact->add_message( $email, 'Email');
  if(isset($_POST['phone'])) {
    $contact->add_message( htmlspecialchars(trim($_POST['phone']), ENT_QUOTES, 'UTF-8'), 'Phone');
  }
  $contact->add_message( $message, 'Message', 10);

  $result = $contact->send();
  if ($result !== 'OK') {
    http_response_code(500);
  }
  echo $result;
?>
