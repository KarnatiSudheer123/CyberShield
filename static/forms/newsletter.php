<?php
  require_once 'form-init.php';

  $contact = create_email_form(
    $default_receiving_email,
    $_POST['email'],
    $_POST['email'],
    "New Subscription: " . $_POST['email']
  );

  $contact->add_message( $_POST['email'], 'Email');

  echo $contact->send();
?>
