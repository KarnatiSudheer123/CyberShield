<?php
/**
 * Shared form initialization utility.
 * Loads the PHP Email Form library and returns a configured form instance.
 *
 * Usage:
 *   require_once 'form-init.php';
 *   $contact = create_email_form($receiving_email, $from_name, $from_email, $subject);
 */

// Replace contact@example.com with your real receiving email address
$default_receiving_email = 'contact@example.com';

function load_email_form_library() {
  $php_email_form = '../assets/vendor/php-email-form/php-email-form.php';
  if (file_exists($php_email_form)) {
    include_once($php_email_form);
  } else {
    die('Unable to load the "PHP Email Form" Library!');
  }
}

function create_email_form($to, $from_name, $from_email, $subject) {
  load_email_form_library();

  $form = new PHP_Email_Form;
  $form->ajax = true;
  $form->to = $to;
  $form->from_name = $from_name;
  $form->from_email = $from_email;
  $form->subject = $subject;

  // Uncomment below code if you want to use SMTP to send emails.
  // You need to enter your correct SMTP credentials.
  /*
  $form->smtp = array(
    'host' => 'example.com',
    'username' => 'example',
    'password' => 'pass',
    'port' => '587'
  );
  */

  return $form;
}
?>
