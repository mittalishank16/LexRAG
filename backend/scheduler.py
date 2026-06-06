from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import date, timedelta
import sendgrid 
from sendgrid.helpers.mail import Mail 
import os 
from db.supabase_client import get_supabase

scheduler = AsyncIOScheduler() 

async def check_contract_renewals(): 
    supabase = get_supabase() 
    today = date.today() 
    
    # Evaluate each milestone sequence independently
    for days_ahead, col in [(30, 'reminder_sent_30'), (7, 'reminder_sent_7'), (1, 'reminder_sent_1')]: 
        target = today + timedelta(days=days_ahead) 
        
        # Removed backslashes (\) to ensure clean execution parsing
        response = supabase.table('contracts') \
            .select('*') \
            .eq('renewal_date', str(target)) \
            .eq(col, False) \
            .execute() 
        
        # Process data inside the window loop to prevent variables from being overwritten
        for contract in response.data: 
            send_reminder_email(contract, days_ahead) 
            
            # Commit update immediately for the accurate column tracking flag
            supabase.table('contracts') \
                .update({col: True}) \
                .eq('id', contract['id']) \
                .execute() 
        

def send_reminder_email(contract: dict, days_ahead: int): 
    sg = sendgrid.SendGridAPIClient(api_key=os.environ['SENDGRID_API_KEY']) 
    msg = Mail( 
        from_email='noreply@lexrag.app', 
        to_emails=contract.get('user_email', 'admin@lexrag.app'), 
        subject=f'Contract Renewal in {days_ahead} Days: {contract.get("contract_type", "Agreement")}', 
        html_content=f''' 
        <h2>Contract Renewal Reminder</h2> 
        <p>Your <strong>{contract.get('contract_type', 'Agreement')}</strong> between 
        {contract.get('party_a', 'N/A')} and {contract.get('party_b', 'N/A')} renews on 
        <strong>{contract.get('renewal_date', 'N/A')}</strong>.</p>
        <p>That is {days_ahead} days away.</p> 
        <p><em>Key obligations:</em> {contract.get('key_obligations', 'N/A')}</p> 
        ''' 
    ) 
    sg.send(msg)

# Register job: check every morning at 8 AM IST 
scheduler.add_job(check_contract_renewals, 'cron', hour=8, minute=0)