<?php
  require_once 'form-init.php';

  $contact = create_email_form(
    $default_receiving_email,
    $_POST['name'],
    $_POST['email'],
    $_POST['subject']
  );

  $contact->add_message( $_POST['name'], 'From');
  $contact->add_message( $_POST['email'], 'Email');
  if(isset($_POST['phone'])) {
    $contact->add_message( $_POST['phone'], 'Phone');
  }
  $contact->add_message( $_POST['message'], 'Message', 10);

  echo $contact->send();
?>
