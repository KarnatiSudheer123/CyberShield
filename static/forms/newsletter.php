<?php
  /**
  * Requires the "PHP Email Form" library
  * The "PHP Email Form" library is available only in the pro version of the template
  * The library should be uploaded to: vendor/php-email-form/php-email-form.php
  * For more info and help: https://bootstrapmade.com/php-email-form/
  */

  // Replace contact@example.com with your real receiving email address
  $receiving_email_address = 'contact@example.com';

  if( file_exists($php_email_form = '../assets/vendor/php-email-form/php-email-form.php' )) {
    include( $php_email_form );
  } else {
    die( 'Unable to load the "PHP Email Form" Library!');
  }

  if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    die('Method Not Allowed');
  }

  $email = isset($_POST['email']) ? filter_var(trim($_POST['email']), FILTER_SANITIZE_EMAIL) : '';

  if (empty($email) || !filter_var($email, FILTER_VALIDATE_EMAIL)) {
    http_response_code(400);
    die('Please provide a valid email address.');
  }

  $contact = new PHP_Email_Form;
  $contact->ajax = true;

  $contact->to = $receiving_email_address;
  $contact->from_name  = $email;
  $contact->from_email = $email;
  $contact->subject    = "New Subscription: " . $email;

  // SMTP credentials should be loaded from environment variables, not hardcoded.
  // Example:
  //   $contact->smtp = array(
  //     'host'     => getenv('SMTP_HOST'),
  //     'username' => getenv('SMTP_USERNAME'),
  //     'password' => getenv('SMTP_PASSWORD'),
  //     'port'     => getenv('SMTP_PORT') ?: '587'
  //   );

  $contact->add_message( $email, 'Email');

  echo $contact->send();
?>
