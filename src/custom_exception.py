import traceback
import sys


class custom_exception(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_messsage=error_message, error_detail=error_detail)
                            
    @staticmethod
    def get_detailed_error_message(error_messsage, error_detail:sys):
        _,_, exc_tb = traceback.sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in script: {file_name} at line number: {line_number} error message: {error_messsage}"
        
        return error_message
    
    def __str__(self):
        return self.error_message