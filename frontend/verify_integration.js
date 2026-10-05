const axios = require('axios');

async function verify() {
  const frontendUrl = 'http://localhost:3000';
  console.log('Testing integration...');

  try {
    // 1. Submit Query
    console.log('1. Submitting query...');
    const queryRes = await axios.post(`${frontendUrl}/api/v1/query`, {
      query: "Test frontend-backend integration"
    });
    
    if (!queryRes.data.success) {
      console.error('Query failed:', queryRes.data);
      return;
    }
    
    const taskId = queryRes.data.data.taskId;
    console.log('Task ID received:', taskId);
    
    // 2. Poll Task
    console.log('2. Polling task...');
    let status = 'pending';
    let retries = 0;
    
    while (status !== 'completed' && retries < 20) {
      await new Promise(r => setTimeout(r, 1000));
      const taskRes = await axios.get(`${frontendUrl}/api/v1/task/${taskId}`);
      
      if (taskRes.data.success) {
        const taskData = taskRes.data.data;
        status = taskData.status;
        console.log(`Status: ${status}`);
        
        if (taskData.timeline && taskData.timeline.length > 0) {
           console.log('Timeline update:', taskData.timeline[taskData.timeline.length - 1].message);
        }
        
        if (status === 'completed') {
          console.log('Task completed!');
          console.log('Result:', taskData.result.answer.substring(0, 50) + '...');
          return;
        }
      }
      retries++;
    }
    
    console.log('Timeout waiting for task completion');
    
  } catch (error) {
    console.error('Error during verification:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
  }
}

verify();