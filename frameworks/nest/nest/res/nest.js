/* Falcon - Nest framework - AJAX support */

var Nest;
if(!Nest) { Nest = {}; }


(function () {
   "use strict";

if (!Array.prototype.indexOf) {
    Array.prototype.indexOf = function (obj, fromIndex) {
        if (fromIndex == null) {
            fromIndex = 0;
        } else if (fromIndex < 0) {
            fromIndex = Math.max(0, this.length + fromIndex);
        }
        for (var i = fromIndex, j = this.length; i < j; i++) {
            if (this[i] === obj)
                return i;
        }
        return -1;
    };
}

   //============================================================================ Private part
   /* array of {func:, obj:, params:, isAjax:, errorCount:} */
   var pendingFuncs = new Array();
   var isDequeueing = false;
   var currentPendingFunc = null;
   
   function onAjaxResult( http, callback, errCallback )
   {
      var err = null;
      
      if(http.readyState == 4 ) {
         if ( http.status == 200 ) {
            var obj;
            try {
               // try to understand as json.
               obj = JSON.parse(http.responseText);
            }
            catch( err ) {
               // don't bother fulfilling other requests.
               Nest.clearAjaxReqs();
               // if not json, raise proper error.
               Nest.onJSONError( http.responseText, err )
            }

            if( obj ) {
               // application error?
               if ( obj.error ) {
                  // don't bother fulfilling other requests.
                  Nest.clearAjaxReqs();
                  Nest.onAPIError( obj );
               }
               else if (callback) {
                  callback( obj );
               }
            }
         }
         else {
            if( currentPendingFunc && currentPendingFunc.errorCount < 2 ) {
               currentPendingFunc.errorCount = currentPendingFunc.errorCount + 1;
               Nest.requeue();
            }
            else {
               if( errCallback ) errCallback( http.status, http.responseText );
               else {
                  // don't bother fulfilling other requests.
                  Nest.clearAjaxReqs();
                  Nest.onAJAXError( http.status, http.responseText );
               }
            }
            
         }
      }

      // continue the suspended dequeue process.
      Nest.dequeue();
   }
   
   function ajax( url, data, callback, errCallback ) {
      var http = window.XMLHttpRequest ? new XMLHttpRequest() : new ActiveXObject("Microsoft.XMLHTTP");
      http.onreadystatechange = function() {onAjaxResult(http, callback, errCallback); };
      
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

      if( data ) {
         http.send(params);
      }
      else {
         http.send(null);
      }
   }
   
   function addAjaxRequest( url_, data_, callback_, errCallback_ )
   {      
      // enqueue function that will invoke ajax() and suspend queued functions handling
      Nest.enqueue(
            function(){ ajax( url_, data_, callback_, errCallback_); return true; },
            null, null, true );
   }

   // Handler for set message
   function handler_set( obj ) {
      var element = document.getElementById( obj.id );
      if( element ) {
         if( obj.property == 'value' && element.setValue ) {
            element.setValue( obj.value )
         }
         else {
            element[obj.property] = obj.value;
         }
      }
   }

   // Handler for set message
   function handler_set_style( obj ) {
      var element = document.getElementById( obj.id );
      if( element ) { element.style[obj.property] = obj.value; }
   }
   
   // Handler for invoke message
   function handler_invoke( obj ) {
      var element = document.getElementById( obj.id );
      if( element ) { element[obj.method].call( element, obj.param ); }
   }
   
   // Handler for call message
   function handler_call( obj ) {
      if( element ) { document[obj.method].call( null, obj.param ); }
   }
   
   function prepareInit( widID, obj ) {
      // get the root widget.
      
      var list = widID.split(".");
      while( list.length > 0 )
      {
         widID = list.join(".");
         var rootElement = document.getElementById( widID );
         // If we have a root widget, see if we assigned an init info to it.
         if ( rootElement && rootElement.Nest_initInfo )
         {
            obj.init = rootElement.Nest_initInfo;
            break;
         }
         list.pop();
      }
   }

   //=================================================================================== Public interace
   // Method 'i' -- shortcut for document.getElementByID
   if (typeof Nest.i !== 'function') {
      Nest.i = function ( id ) { return document.getElementById( id ); }
   }

   // Stop event propagation
   Nest.eatEvent = function(evt){
      if(typeof(event) != "undefined") event.cancelBubble = true;
      else evt.stopPropagation();
   }
   
   // All the widgets declared by nest
   if (!Nest.w) { Nest.w = new Array();}

   // Method 'rw' -- relative widget.
   if (typeof Nest.rw !== 'function') {
      Nest.rw = function ( wid, path ) {
         var pathArr = path.split("/")
         var widIdArr = wid.id.split( "." );
         while( pathArr.length > 0 && pathArr[0] == '..' ) {
            widIdArr.pop();
            pathArr.shift();
         }
         widIdArr = widIdArr.concat( pathArr );
         var widID = widIdArr.join( "." );
         wid = document.getElementById( widID );
         return wid;
      }
   }

   // Method rp - relative property
   // Get a property in a relative widget.
   // info: the live property to be taken.
   // widID: optional entity ID to which we are relative
   // objName: optional; if given will be filled with:
   //  - name: the value of the "name" property in the target item.
   //  - id: document ID of the target property
   //  - path: relative path in info, transformed into local ID
   // Returns the value of the property.
   //
   Nest.rp = function( info, widID, objName ) {
      var element;
      var useName = false;

      var idArr = info.split("/");
      var valname = idArr.pop();

      if( idArr.length == 0 && widID != null ) {
         element = document.getElementById( widID );
      }
      else {
         if( idArr[0] == "" ) {
            // the path is absolute
            idArr.shift();
            element = document.getElementById( idArr.join(".") );
         }
         else {
            var widIdArr = widID == null? [] : widID.split( "." );
            while( idArr.length > 0 && idArr[0] == '..' ) {
               widIdArr.pop();
               idArr.shift();
            }
            widIdArr = widIdArr.concat( idArr );
            var wid = widIdArr.join( "." );
            element = document.getElementById( wid );
         }
      }

      if (element != null) {
         // get the proper value
         var value;
         if ( valname == 'value' && element.getValue != null )
            value = element.getValue();
         else
            value = element[valname];

         // recreate the full entity name, re-localized after .. purging.
         // if the object was given, we want name, property name and path name
         if( objName != null ) {
            var name = element.name;
            if( name ) objName['name'] = name.replace("[]", "")
            objName['id'] = element.id;
            // empty if we didn't pop ../
            idArr.push( valname );
            objName['path'] = idArr.join(".");
         }

         return value;
      }
   }

   // Method 'ajax' -- enqueues a request to perform an AJAX query.
   if (typeof Nest.ajax !== 'function') {
      Nest.ajax = function ( req_id, params, callback ) {
         var url = "./?a=" + req_id;
         addAjaxRequest( url, params, callback );
      }
   }
   
   // Method 'enqueue' -- puts a function in a queue and execute when dequeue is invoked.
   // -- If the enqueue function returns true, dequeueing is suspended and then Nest.dequeue()
   // -- must be explicitly called again to continue the process.
   if (typeof Nest.enqueue !== 'function') {
      Nest.enqueue = function ( func_, params_, obj_, isAjax_ ) {
         pendingFuncs.push( {func: func_, params: params_, obj: obj_, isAjax: isAjax_, errCount:0 } );
         this.dequeue();
      }
   }
   
   // Method 'posticipate' -- Like enequeue, but doesn't statr a dequeue automatically.
   if (typeof Nest.posticipate !== 'function') {
      Nest.posticipate = function ( func_, params_, obj_, isAjax_ ) {
         pendingFuncs.push( {func: func_, params: params_, obj: obj_, isAjax: isAjax_, errCount:0 } );
      }
   }

   // Method 'dequeue' -- starts executing functions in queue; does nothing is already dequeueing.
   if (typeof Nest.dequeue !== 'function') {
      Nest.dequeue = function ( func_, params_, obj_, isAjax_ ) {
         if( isDequeueing ) {return;}
         
         isDequeueing = true;
         var suspend =  false;
         while( pendingFuncs.length > 0 ) {
            currentPendingFunc = pendingFuncs.shift();
            suspend = currentPendingFunc.func.call( currentPendingFunc.obj, currentPendingFunc.params );
            if( suspend ) { break; }
         }
         // if we're suspended, we must keep currentPendingFunc, else we must clear it
         if( ! suspend ) currentPendingFunc = null;
         isDequeueing = false;
      }
   }
   
   // Method 'requeue' -- resubmits immediately the queued function that is currently being executed.
   if (typeof Nest.requeue !== 'function') {
      Nest.requeue = function ( func_, params_, obj_, isAjax_ ) {
         if( currentPendingFunc ) {
            pendingFuncs.unshift( currentPendingFunc );
         }
      }
   }
   
   //Method clearAjaxReqs -- removes queued ajax requests, keeping other queued functions.
   if (typeof Nest.clearAjaxReqs !== 'function') {
      Nest.clearAjaxReqs = function() {
         var tempArr = new Array();
         for( var i in pendingFuncs ) {
            var req = pendingFuncs[i];
            if( ! req.isAjax ) {
               tempArr.push(req);
            }
         }
         
         pendingFuncs = tempArr;
      }
   }
   
   //Method clearQueued -- removes all the queued functions.
   if (typeof Nest.clearQueued !== 'function') {
      Nest.clearQueued = function() {
         pendingFuncs = new Array();
      }
   }
   
   if (typeof Nest.setWidVal !== 'function') {
      Nest.setWidVal = function ( wid, value ) {
         var element = Nest.i(wid);
         if( element != null ) {
            if( element.setValue != null ) {
               element.setValue( value );
            }
            else {
               element.value = value;
            }
         }         
      }
   }

   if (typeof Nest.makeproto !== 'function') {
      Nest.makeproto = function ( proto, target ) {
         var f;
         for (f in proto) {
            target[f] = proto[f];
         }
      }
   }

   // Method 'widgetMsg' -- Sending AJAX requests to remote widget server.
   if (typeof Nest.widgetAJAX !== 'function') {
      Nest.widgetAJAX = function ( widClass, widID, msg, params ) {
         // the object to be sent.
         var objToSend = { 'widget': widClass, 'id': widID, 'msg': msg };
         // let's get rid of the details now -- this is extra data to send as-is
         prepareInit( widID, objToSend );
         
         // Params
         if( params != null) {
            objToSend["params"] = params;
         }

         var url = "./?w=" + widClass;
         //alert( JSON.stringify( objToSend ) );
         addAjaxRequest( url, objToSend, Nest.widgetUpdate );
      }
   }

   // Method 'message' -- sends a local message to listeners in the page.
   if (typeof Nest.message !== 'function') {
      Nest.message = function ( wid, msg, value ) {
         // enqueue the whole processing, so that a subscribe in front of the queue is correctly honoured.
         Nest.enqueue( function() {
            var listener = Nest.listeners[wid.id];
            if( listener ) {
               for (var i = 0; i < listener.length; i++) {
                  var lrec = listener[i];
                  var func = lrec.func;
                  var tgtid = lrec.tgt;
                  
                  func.call( tgtid, wid, msg, value );
               }
            }
         } );
      }
   }

   // Method 'listen' -- Waits for updates on a certain widget ID
   // callbacks are in this prototype: func( target, source_wid, msg, value );
   if (typeof Nest.listen !== 'function') {
      Nest.listen = function ( target, wid, cbfunc ) {
         Nest.enqueue( function() {
            var listener = Nest.listeners[wid];
            var listenRecord = { "tgt": target, "func": cbfunc };
            if( listener ) {
               listener.push( listenRecord );
            }
            else {
               Nest.listeners[wid] = Array( listenRecord );
            }
         } );
      } 
   }   
   
   
   // Method 'widgetUpdate' -- handling requests from widget server.
   if (typeof Nest.widgetUpdate !== 'function') {
      Nest.widgetUpdate = function ( obj ) {
         Nest.enqueue( function() {
            // handle multiple messages.
            if( typeof obj == 'object' && obj.length ) {
               var i = 0;
               while( i < obj.length ) { Nest.processMessage( obj[i] ); i = i + 1; }
            }
            else {
               Nest.processMessage( obj );
            }
         });
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
               Nest.enqueue( function() {
                  handler.method.call( handler.object, obj );
               });
            }
         }
         else {
            Nest.onWidgetUpdateError( obj );
         }
      }
   }

   // Method 'processMessage' -- handling a single request from widget server.
   if (typeof Nest.listenAJAX !== 'function') {
      Nest.listenAJAX = function ( msg, obj, func ) {        
         Nest.messageHandlers[ msg ] = { object: obj, method: func };
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
   //=========================================================================== DynParams API
   if (typeof Nest.startPar !== 'function') {
      Nest.startPar = function(wid) {
         return {
               params: {},
               add: function( key, value ) {
                     this.params[key] = value; return this; },
               addPath: function( key, value ) {
                     this.params[key] = Nest.rp(value,wid); return this; },
               addTPath: function( value ) {
                     var names = {};
                     var v = Nest.rp(value,wid,names);
                     this.params[names.path] = v;
                     return this; },
               addName: function( value ) {
                     var names = {};
                     var v = Nest.rp(value,wid,names);
                     this.params[names.name] = v;
                     return this;
                  },
               addId: function( value ) {
                     var names = {};
                     var v = Nest.rp(value,wid,names);
                     this.params[names.id] = v;
                     return this;
                  },
               gen: function() {
                  return this.params; }
         }
      }
   }
   

   //=========================================================================== Object initialization.
   // set the default widget server message handlers
   if (! Nest.messageHandlers ) {
      Nest.messageHandlers = {
         'set': { object: null, method: handler_set},
         'set_style': { object: null, method: handler_set_style},
         'invoke': { object: null, method: handler_invoke},
         'call': { object: null, method: handler_call}
      }
   }

   // Set the local message handlers
   if (! Nest.listeners ) {
      Nest.listeners = {};
   }

}());


/* http://www.JSON.org/json2.js -- See http://www.JSON.org/js.html */
var JSON;
if (!JSON) {
    JSON = {};
}
(function () {
    "use strict";

    function f(n) {
        // Format integers to have at least two digits.
        return n < 10 ? '0' + n : n;
    }

    if (typeof Date.prototype.toJSON !== 'function') {

        Date.prototype.toJSON = function (key) {

            return isFinite(this.valueOf()) ?
                this.getUTCFullYear() + '-' +
                f(this.getUTCMonth() + 1) + '-' +
                f(this.getUTCDate()) + 'T' +
                f(this.getUTCHours()) + ':' +
                f(this.getUTCMinutes()) + ':' +
                f(this.getUTCSeconds()) + 'Z' : null;
        };

        String.prototype.toJSON =
            Number.prototype.toJSON =
            Boolean.prototype.toJSON = function (key) {
                return this.valueOf();
            };
    }

    var cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
        escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
        gap,
        indent,
        meta = { // table of character substitutions
            '\b': '\\b',
            '\t': '\\t',
            '\n': '\\n',
            '\f': '\\f',
            '\r': '\\r',
            '"' : '\\"',
            '\\': '\\\\'
        },
        rep;


    function quote(string) {
        escapable.lastIndex = 0;
        return escapable.test(string) ? '"' + string.replace(escapable, function (a) {
            var c = meta[a];
            return typeof c === 'string' ? c :
                '\\u' + ('0000' + a.charCodeAt(0).toString(16)).slice(-4);
        }) + '"' : '"' + string + '"';
    }


    function str(key, holder) {
        var i, // The loop counter.
            k, // The member key.
            v, // The member value.
            length,
            mind = gap,
            partial,
            value = holder[key];
        if (value && typeof value === 'object' &&
                typeof value.toJSON === 'function') {
            value = value.toJSON(key);
        }
        if (typeof rep === 'function') {
            value = rep.call(holder, key, value);
        }
        switch (typeof value) {
        case 'string':
            return quote(value);

        case 'number':
            return isFinite(value) ? String(value) : 'null';

        case 'boolean':
        case 'null':
            return String(value);
        case 'object':
            if (!value) {
                return 'null';
            }
            gap += indent;
            partial = [];
            if (Object.prototype.toString.apply(value) === '[object Array]') {
                length = value.length;
                for (i = 0; i < length; i += 1) {
                    partial[i] = str(i, value) || 'null';
                }
                v = partial.length === 0 ? '[]' : gap ?
                    '[\n' + gap + partial.join(',\n' + gap) + '\n' + mind + ']' :
                    '[' + partial.join(',') + ']';
                gap = mind;
                return v;
            }

            if (rep && typeof rep === 'object') {
                length = rep.length;
                for (i = 0; i < length; i += 1) {
                    if (typeof rep[i] === 'string') {
                        k = rep[i];
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ': ' : ':') + v);
                        }
                    }
                }
            } else {
                for (k in value) {
                    if (Object.prototype.hasOwnProperty.call(value, k)) {
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ': ' : ':') + v);
                        }
                    }
                }
            }
            v = partial.length === 0 ? '{}' : gap ?
                '{\n' + gap + partial.join(',\n' + gap) + '\n' + mind + '}' :
                '{' + partial.join(',') + '}';
            gap = mind;
            return v;
        }
    }
    if (typeof JSON.stringify !== 'function') {
        JSON.stringify = function (value, replacer, space) {
            var i;
            gap = '';
            indent = '';
            if (typeof space === 'number') {
                for (i = 0; i < space; i += 1) {
                    indent += ' ';
                }
            } else if (typeof space === 'string') {
                indent = space;
            }
            rep = replacer;
            if (replacer && typeof replacer !== 'function' &&
                    (typeof replacer !== 'object' ||
                    typeof replacer.length !== 'number')) {
                throw new Error('JSON.stringify');
            }
            return str('', {'': value});
        };
    }
    if (typeof JSON.parse !== 'function') {
        JSON.parse = function (text, reviver) {
            var j;

            function walk(holder, key) {
                var k, v, value = holder[key];
                if (value && typeof value === 'object') {
                    for (k in value) {
                        if (Object.prototype.hasOwnProperty.call(value, k)) {
                            v = walk(value, k);
                            if (v !== undefined) {
                                value[k] = v;
                            } else {
                                delete value[k];
                            }
                        }
                    }
                }
                return reviver.call(holder, key, value);
            }
            text = String(text);
            cx.lastIndex = 0;
            if (cx.test(text)) {
                text = text.replace(cx, function (a) {
                    return '\\u' +
                        ('0000' + a.charCodeAt(0).toString(16)).slice(-4);
                });
            }
            if (/^[\],:{}\s]*$/
                    .test(text.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, '@')
                        .replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, ']')
                        .replace(/(?:^|:|,)(?:\s*\[)+/g, ''))) {
                j = eval('(' + text + ')');
                return typeof reviver === 'function' ?
                    walk({'': j}, '') : j;
            }
            throw new SyntaxError('JSON.parse');
        };
    }
}());

