 $(function(){
        csmapi.set_endpoint ('https://6.iottalk.tw');
        var profile = {
		    'dm_name': 'awesome_animal',          
			'idf_list':[predict_animal],
			'odf_list':[to_organizer],
		    'd_name': 'test',
        };

        setInterval(predict_animal , 3000);
        
        function predict_animal(){
            var ans = document.getElementById("ans").innerHTML;
            return ans;
        }    

        var old_data = "";

        function to_organizer(data){
            var new_data = data[0];
            if(new_data !== old_data){
                console.log("ans: ", new_data);
                old_data = new_data;
            }
        }
      
/*******************************************************************/                
        function ida_init(){
	    console.log(profile.d_name);
	}
        var ida = {
            'ida_init': ida_init,
        }; 
        dai(profile,ida);     
});
