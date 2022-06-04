 $(function(){
        //csmapi.set_endpoint ('https://6.iottalk.tw');
        var profile = {
		    'dm_name': 'awesome_animal',          
			//'idf_list':[Dummy_Sensor],
			'odf_list':[to_organizer],
		    'd_name': 'test_out',
        };
		
        // function Dummy_Sensor(){
        //     return Math.random();
        // }

        var old_data = "";

        function to_organizer(data){
           var get_animal = data[0];

            if(get_animal !== old_data && get_animal !== "waiting picture"){
                console.log("ans: ", get_animal);
                // change html
                document.getElementById(get_animal).removeAttribute("style");
                if(get_animal == "dog" || get_animal == "cat"){
                    document.getElementById(get_animal).href = get_animal+"_detail.html";
                }

                old_data = get_animal;
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
