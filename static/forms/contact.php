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

  $name    = isset($_POST['name'])    ? htmlspecialchars(strip_tags(trim($_POST['name'])), ENT_QUOTES, 'UTF-8')    : '';
  $email   = isset($_POST['email'])   ? filter_var(trim($_POST['email']), FILTER_SANITIZE_EMAIL)                   : '';
  $subject = isset($_POST['subject']) ? htmlspecialchars(strip_tags(trim($_POST['subject'])), ENT_QUOTES, 'UTF-8') : '';
  $message = isset($_POST['message']) ? htmlspecialchars(strip_tags(trim($_POST['message'])), ENT_QUOTES, 'UTF-8') : '';
  $phone   = isset($_POST['phone'])   ? htmlspecialchars(strip_tags(trim($_POST['phone'])), ENT_QUOTES, 'UTF-8')   : '';

  if (empty($name) || empty($email) || empty($subject) || empty($message)) {
    http_response_code(400);
    die('Please fill in all required fields.');
  }

  if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    http_response_code(400);
    die('Please provide a valid email address.');
  }

  $contact = new PHP_Email_Form;
  $contact->ajax = true;

  $contact->to = $receiving_email_address;
  $contact->from_name  = $name;
  $contact->from_email = $email;
  $contact->subject    = $subject;

  // SMTP credentials should be loaded from environment variables, not hardcoded.
  // Example:
  //   $contact->smtp = array(
  //     'host'     => getenv('SMTP_HOST'),
  //     'username' => getenv('SMTP_USERNAME'),
  //     'password' => getenv('SMTP_PASSWORD'),
  //     'port'     => getenv('SMTP_PORT') ?: '587'
  //   );

  $contact->add_message( $name,    'From');
  $contact->add_message( $email,   'Email');
  if (!empty($phone)) {
    $contact->add_message( $phone, 'Phone');
  }
  $contact->add_message( $message, 'Message', 10);

  echo $contact->send();
?>
