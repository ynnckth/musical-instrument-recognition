import smtplib


class EmailSender:
    def __init__(self, config):
        self.save_plots = config.get('save_plots', False)
        self.experiment_name = config.get('experiment_name', 'default')

    # The call method will be called from nolearn when all epochs trained
    def __call__(self, nn, train_history):
        if self.save_plots:
            from_address = 'julian_schmid@gmx.ch'
            to_address = ['leo.zulfiu@gmx.ch', "yannickstreit@hotmail.com"]

            subject = "experiment finished: {0}".format(self.experiment_name)
            text = "Experiment finished!\n see plots in result dir"

            message = ("From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n"
                       % (from_address, to_address, subject))

            message += text

            server = smtplib.SMTP_SSL('smtp.gmx.ch', 465)
            server.set_debuglevel(False)
            server.ehlo()
            server.login('julian_schmid@gmx.ch', 'd-9e_EDB')

            server.sendmail(from_address, to_address, message)

            server.quit()
            print "Email sent"
