/* Falcon - Nest framework - AJAX support */

var Nest;
if(!Nest) { Nest = {}; }

(function () {
   "use strict";

   //============================================================================ Private part
   function ajax( url, data, callback ) {
      var http = window.XMLHttpRequest ? new XMLHttpRequest() : new ActiveXObject("Microsoft.XMLHTTP");

      var params = "";
      if(data) {
         /*
         for( var key in data )
         {
            if ( params != "" ) { params = params + "&"; }
            params = params + encodeURIComponent(key) + "=" + encodeURIComponent(data[key]);
         }
         */
         params = "params=" + encodeURIComponent( JSON.stringify( data ) );

         http.open("POST", url, true);

         //Send the proper header information along with the request
         http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
         // http.setRequestHeader("Content-length", params.length);
         // http.setRequestHeader("Connection", "close");
      }
      else {
         http.open("GET", url, true);
      }

      http.onreadystatechange = function() { //Call a function when the state changes.
         if(http.readyState == 4 ) {
            if ( http.status == 200 ) {
               var obj;
               try {
                  // try to understand as json.
                  obj = JSON.parse(http.responseText);
               }
               catch( err ) {
                  // if not json, raise proper error.
                  Nest.onJSONError( http.responseText, err )
               }

               if( obj ) {
                  // application error?
                  if ( obj.error ) { Nest.onAPIError( obj ); }
                  else { callback( obj ); }
               }
            }
            else {
               Nest.onAJAXError( http.status, http.responseText );
            }
         }
      }

      if( data ) {
         http.send(params);
      }
      else {
         http.send(null);
      }
   }

   // prepare the information to be sent.
   function prepareInfos( widID, obj, infosToSend ) {
      var i = 0;
      var element;
      while( i < infosToSend.length ) {
         var info = infosToSend[i];
         var idArr = info.split("/");
         var valname = idArr.pop();

         if( idArr.length == 0 ) {
            element = document.getElementById( widID );
         }
         else {
            if( idArr[0] == "" ) {
               // the path is absolute
               idArr.shift();
               element = document.getElementById( idArr.join(".") );
            }
            else {
               var widIdArr = widID.split( "." );
               while( idArr.length > 0 && idArr[0] == '..' ) {
                  widIdArr.pop();
                  idArr.shift();
               }
               widIdArr = widIdArr.concat( idArr );
               var wid = widIdArr.join( "." );
               element = document.getElementById( wid );
            }
         }
         
         if (element[valname] != null) {
            // recreate the full entity name, re-localized after .. purging.
            idArr.push( valname );
            obj[idArr.join(".")] = element[valname];
         }

         i = i + 1;
      }
   }

   // Handler for set message
   function handler_set( obj ) {
      var element = document.getElementById( obj.id );
      if( element ) { element[obj.property] = obj.value; }
   }

   // Handler for invoke message
   function handler_invoke( obj ) {
      var element = document.getElementById( obj.id );
      if( element ) { element[obj.method].call( element, obj.param ); }
   }

   function prepareInit( widID, obj ) {
      // get the root widget.
      var rootID = widID.split(".")[0];
      var rootElement = document.getElementById( rootID );
      // If we have a root widget, see if we assigned an init info to it.
      if ( rootElement && rootElement.Nest_initInfo )
      {
         obj.init = rootElement.Nest_initInfo;
      }
   }

   //=================================================================================== Public interace

   // Method 'ajax'
   if (typeof Nest.ajax !== 'function') {
      Nest.ajax = function ( req_id, params, callback ) {
         var url = "./?a=" + req_id;
         ajax( url, params, callback );
      }
   }

   // Method 'widgetMsg' -- Sending AJAX requests to remote widget server.
   if (typeof Nest.widgetMsg !== 'function') {
      Nest.widgetMsg = function ( widClass, widID, msg, infosToSend, extraParams ) {
         // the object to be sent.
         var objToSend = { 'widget': widClass, 'id': widID, 'msg': msg };
         // let's get rid of the details now -- this is extra data to send as-is
         if( extraParams ) {objToSend.extra = extraParams;}
         prepareInit( widID, objToSend );
         
         // read the infos and store them away
         if( infosToSend ) {
            var infos = {};
            prepareInfos( widID, infos, infosToSend );
            objToSend["infos"] = infos;
         }

         var url = "./?w=" + widClass;
         //alert( JSON.stringify( objToSend ) );
         ajax( url, objToSend, Nest.widgetUpdate );
      }
   }

   // Method 'message' -- sends a local message to listeners in the page.
   if (typeof Nest.message !== 'function') {
      Nest.message = function ( wid, msg, value ) {
         var listener = Nest.listeners[wid];
         if( listener ) {
            for (var i = 0; i < listener.length; i++) {
               var func = listener[i].func;
               var tgtid = listener[i].tgt;
               func.call( tgtid, wid, msg, value );
            }
         }
      }
   }

   // Method 'listen' -- Waits for updates on a certain widget ID
   // callbacks are in this prototype: func( target, source_wid, msg, value );
   if (typeof Nest.listen !== 'function') {
      Nest.listen = function ( target, wid, cbfunc ) {
         var listener = Nest.listeners[wid];
         var listenRecord = { "tgt": target, "func": cbfunc };
         if( listener ) {
            listener.push( listenRecord );
         }
         else {
            Nest.listeners[wid] = Array( listenRecord );
         }
      }
   }   
   
   
   // Method 'widgetUpdate' -- handling requests from widget server.
   if (typeof Nest.widgetUpdate !== 'function') {
      Nest.widgetUpdate = function ( obj ) {
         // handle multiple messages.
         if( typeof obj == 'object' && obj.length ) {
            var i = 0;
            while( i < obj.length ) { Nest.processMessage( obj[i] ); i = i + 1; }
         }
         else {
            Nest.processMessage( obj );
         }
      }
   }

   // Method 'processMessage' -- handling a single request from widget server.
   if (typeof Nest.processMessage !== 'function') {
      Nest.processMessage = function ( obj ) {
         if( obj.message ) {
            var handler = Nest.messageHandlers[ obj.message ];
            if( ! handler ) {
               Nest.onMessageNotFound( obj );
            }
            else {
               handler( obj );
            }
         }
         else {
            Nest.onWidgetUpdateError( obj );
         }
      }
   }
   
   //=========================================================================== Error management.
   if (typeof Nest.onAJAXError !== 'function') {
      Nest.onAJAXError = function( code, text ){
         alert( "Nest framework AJAX error.\n" +
            "Response from server: " + code + "\n" +
            text );
      }
   }

   if (typeof Nest.onJSONError !== 'function') {
      Nest.onJSONError = function( text, synerr ){
         alert( "Nest framework AJAX error.\n" +
            "JSON parse error: " + synerr.name+"\n"+ synerr.message +"\n" +
            "Response was: " + text
            );
      }
   }

   if (typeof Nest.onAPIError !== 'function') {
      Nest.onAPIError = function( obj ){
         alert( "Nest framework AJAX API error.\n" +
            "Remote API error : " + obj.error +"\n"+
               obj.errorDesc
            );
      }
   }

    if (typeof Nest.onMessageNotFound !== 'function') {
      Nest.onMessageNotFound = function( obj ){
         alert( "No handler registered for Nest widget message '" + obj.message + "'.\n" +
            "Received: " + obj +"\n"
            );
      }
   }

    if (typeof Nest.onWidgetUpdateError !== 'function') {
      Nest.onWidgetUpdateError = function( obj ) {
         alert( "Not a widget update message in Nest widget update.\n" +
            "Received: " + obj +"\n"
            );
      }
   }

   //=========================================================================== Object initialization.
   // set the default widget server message handlers
   if (! Nest.messageHandlers ) {
      Nest.messageHandlers = {
         'set': handler_set,
         'invoke': handler_invoke
      }
   }

   // Set the local message handlers
   if (! Nest.listeners ) {
      Nest.listeners = {};
   }

}());
