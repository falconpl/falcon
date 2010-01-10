/*
   FALCON - The Falcon Programming Language.
   FILE: cmdlineparser.cpp

   The command line parser class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-11-30

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The command line parser class
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/carray.h>
#include <falcon/vm.h>
#include <falcon/stream.h>
#include "core_module.h"
#include <falcon/eng_messages.h>

/*#
   @class CmdlineParser
   @brief Provides simple and powerful parsing of the command line options.

   Command line options are the simplest and most immediate mean to provide a
   stand-alone script with basic configuration, or to ask it to do something a
   bit more specific than just "operate".

   Some embedding applications may provide the scripts with a command line too;
   for example, a "scanner" script in a FPS game may be provided with the objects
   to search for in a "command line", that may be actually the string that represents
   its configuration in the user interface.

   Often, this important feature is neglected in scripts because bringing up a decent
   option parser is a bit of a nuisance, boring and repetitive, and above anything
   it may be considered a huge piece of code with respect to the needs of a simple script.

   The CmdlineParser class, that is declared directly in the RTL module, provides a simple,
   efficient and flexible mean to implement command line option parsing that let on the
   script the essential duty to grab the given values and store them for later usage.

   The command line parser knows the following option categories:

   - @b short @b options: options consisting of a single character, case sensitive,
     following a single "-". For example, "-a", "-B", "-x". Short options may be
     chained in sequences of characters as, for example "-aBx" which is
     equivalent to "-a -B -x". Short options may have also the special
     "switch off" semantic; if they are followed by a "-" sign, the parser
     interprets it as a will to turn off some feature; for example, the
     sequence "-B-" means that the "B" option should be turned off.
     The semantic can be expressed also in chained options as "-aB-x".

   - @b long @b options: they consists of two minus followed by a word of any
     length, as for example "--parameter", "--config", "--elements".
     Long options are usually (but not necessarily) meant to receive a parameter,
     for example "--debug off".

   - @b free @b options: they are strings not leaded by any "-" sign. Usually the semantic
     of a command gives free options a special meaning; for example the "cp" unix command
     accept an arbitrary amount of free options, where the first N-1 options are
     the name of the files to copy, and the Nth option is the copy destination.
     A single "-" not followed by any letter is considered a free option
     (i.e. it often means "stdout/stdin" in many UNIX commands).

   - @b option @b parsing @b terminator: The special sequence "--" followed by a
     whitespace is considered as the terminator of option parsing; after that element,
     all the other options are considered free options and given to the parser "as is".
     If you want to pass a free parameter starting with a "-", (i.e. a file named
     "-strangename"), it must follow the "--" sign.

   Short and long options may be parametric. The word (or string) following parametric
   option is considered the parameter of that option, and is not subject to parsing.
   For example, if "--terminator" is a parametric option, it is possible to write
   "./myprg.fal --terminator -opt". The parameter "-opt" will be passed as-is to the
   script as "terminator" option parameter. In case of short option chaining,
   if more than one chained option is parametric, the parameter following the chained
   options will be considered applied only to the last option, and the other ones
   will be ignored. If the sequence of parameters ends while waiting for the parameter
   of an option, the incomplete option is ignored.

   On the script point of view, the parser can be configured by implementing callbacks
   in the CmdlineParser class. The parser will call the methods of the subclasses as
   it finds options in the argument vector; the callbacks will configure the application,
   report errors and mis-usage, terminate the program on fatal errors and communicate
   with the parser through member functions. For example, it is not necessary to declare
   in advance which are the parametric options; it's done on a per-option basis by calling
   the expectParam() method and returning to the parser.

   To use this feature, it is just necessary to declare a subclass of CmdlineParser and
   instance it, or derive an object from it, and call the parse() method.

   The CmdlineParser class is meant to be overloaded by scripts, so that the
   callbacks provided in the class can be called by the parse() method. Once
   called, parse() will scan the line and will call in turn onOption(), onFree()
   and onSwitchOff() callbacks, depending on what kind of arguments it finds in
   the argument list. If the parsed option should provide some value, the script
   implementing onOption() should call expectValue() and then return. The next
   element on the command line will be then passed to onValue(), which will receive
   the previously parsed option and the parsed value as parameters.

   Calling the terminate() method from any callback routine will force parse() to
   terminate and pass the control back to the application. The last parsed element
   will be stored in the property lastParsed, that can be used as an index to read
   the args vector from the last parsed parameter.

   Here is a working sample of implementation.

   @code
   object MyParser from CmdlineParser

      function onOption( option )
         switch option
            case "?", "help"
               self.usage()
            case "d", "m", "l", "long"
               // those options require a parameter; signal it
               self.expectValue()
            case "k"
               // set switch k ON
            case "n"
               // set switch n ON
            case "v", "version"
               // show version
            case "z", "sleep"
               self.terminate()
            default
               self.unrecognized( option )
         end
      end

      function onValue( option, value )
         switch option
            case "d"
               // set value for option d
            case "m"
               // set value for option m
            case "l", "long"
               // set value for option l
         end
         // can't be anything else, as this function call must
         // be authorized from onOption
      end

      function onFree( param )
         // record the free parameter
      end

      function onSwitchOff( sw )
         switch sw
            case "k"
               // set switch k OFF
            case "n"
               // set switch n OFF
            default
               self.unrecognized( sw )
         end
      end

      function unrecognized( option )
         // Signal some error
      end

      function usage()
         // say something relevant
      end
   end

   // And in the main program:
   MyParser.parse()
   @endcode

   @b Notice: Callback methods in the instance are called in Virtual Machine atomic mode.
   The called methods cannot be interrupted by external kind requests, they won't honor
   periodic callback requests and and they will be forbidden to sleep or yield the
   execution to other coroutines. Parsing of the whole command line happens
   in an atomic context, so it's not possible to wait for other coroutines in
   anyone of the callback methods. It is also advisable that methods are simple
   and straight to the point to minimize the time in which the VM is unresponsive
   to kind requests and time scheduling.

   @prop lastParsed An integer representing the last item parsed in the argument
          array before exit the parsing loop.


*/

namespace Falcon{
namespace core {

/*#
   @method parse CmdlineParser
   @brief Starts command line parsing.
   @optparam args A specific string array that will be used as arguments to be parsed.
   @return true if the parsing is complete, false on error.

   Start the parsing process. If args parameter is not provided, the method gets
   the content of the @a args global vector defined in the Core module.

   Returns true if the parsing was complete, and false on error (for example,
   if some element in the array wasn't a string).
*/
FALCON_FUNC  CmdlineParser_parse( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_params = vm->param( 0 );

   if ( i_params == 0 )
   {
      // get the parameters from the VM args object
      i_params = vm->findGlobalItem( "args" );
      if ( i_params == 0 || ! i_params->isArray() ) {
         throw new CodeError( ErrorParam( e_undef_sym ).extra( "args" ).hard() );
      }
   }
   else if ( ! i_params->isArray() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).extra( "( A )" ) );
   }

   CoreArray *args = i_params->asArray();

   // zero request.
   self->setProperty( "_request", Item((int64) 0) );
   self->setProperty( "lastParsed", Item((int64) 0) );

   // status.
   typedef enum {
      t_none,
      t_waitingValue,
      t_allFree
   } t_states;

   t_states state = t_none ;
   String currentOption;
   Item i_method;
   Item i_passMM;
   self->getProperty( "passMinusMinus", i_passMM );
   bool passMM = i_passMM.isTrue();
   Item _request;
   String subParam;
   uint32 i;

   for ( i = 0; i < args->length(); i++ )
   {
      Item &i_opt = args->at( i );
      if ( !i_opt.isString() )
      {
         throw new ParamError( ErrorParam( e_param_type ).
                  extra( vm->moduleString( rtl_cmdp_0 ) ) );
      }

      String &opt = *i_opt.asString();
       // if we were expecting a value, we MUST consider ANYTHING as it was a value.
      if ( state == t_waitingValue )
      {
         self->getProperty( "onValue", i_method );
         if ( i_method.methodize( self ) )
         {
            vm->pushParam( new CoreString(currentOption) );
            vm->pushParam( i_opt );
            vm->callItemAtomic( i_method, 2 );
            state = t_none;
         }
         else
         {
            vm->retval( false );
            self->setProperty( "lastParsed", i );
            return;
         }
      }
      else if( opt.length() == 0 || (opt.getCharAt( 0 ) != '-' || opt.length() == 1) || state == t_allFree )
      {

         self->getProperty( "onFree", i_method );
         if ( i_method.methodize( self ) )
         {
            vm->pushParam( i_opt );
            vm->callItemAtomic( i_method, 1 );
         }
         else
         {
            vm->retval( false );
            self->setProperty( "lastParsed", i );
            return;
         }
      }
      else if ( opt == "--" && ! passMM )
      {
         state = t_allFree;
         continue; // to skip return value.
      }
      else {
         // we have at least one '-', and length > 1
         if ( opt.getCharAt( 1 ) == (uint32) '-' )
         {
            self->getProperty( "onOption", i_method );

            if ( i_method.methodize( self ) )
            {
               if ( passMM && opt.size() == 2 )
                  vm->pushParam( i_opt );
               else {
                  //Minimal optimization; reuse the same string and memory
                  subParam = opt.subString( 2 );
                  vm->pushParam( new CoreString( subParam ) );
               }

               vm->callItemAtomic( i_method, 1 );
               self->getProperty( "_request", _request );
               // value requested?
               if ( _request.asInteger() == 1 ) {
                  currentOption = subParam;
               }
            }
            else
            {
               vm->retval( false );
               self->setProperty( "lastParsed", i );
               return;
            }
         }
         else {
            // we have a switch set.
            for( uint32 chNum = 1; chNum < opt.length(); chNum++ )
            {
               //Minimal optimization; reuse the same string and memory

               subParam.size( 0 );
               subParam.append( opt.getCharAt( chNum ) );

               if ( chNum < opt.length() -1 && opt.getCharAt( chNum +1 ) == (uint32) '-' )
               {
                  // switch turnoff.
                  self->getProperty( "onSwitchOff", i_method );
                  if ( i_method.methodize( self ) )
                  {
                     vm->pushParam( new CoreString(subParam) );
                     vm->callItemAtomic( i_method, 1 );
                 }
                  else
                  {
                     vm->retval( false );
                     self->setProperty( "lastParsed", (int64) i );
                     return;
                  }
                  chNum++;
               }
               else {
                  self->getProperty( "onOption", i_method );
                  if ( i_method.methodize( self ) )
                  {
                     vm->pushParam( new CoreString(subParam) );
                     vm->callItemAtomic( i_method, 1 );
                  }
                  else
                  {
                     vm->retval( false );
                     self->setProperty( "lastParsed", (int64) i );
                     return;
                  }
               }

               self->getProperty( "_request", _request );
               // value requested?
               if ( _request.asInteger() == 1 ) {
                  currentOption = subParam;
               }
            }
         }

         self->getProperty( "_request", _request );
         // value requested?
         if ( _request.asInteger() == 1 ) {
            state = t_waitingValue;
            self->setProperty( "_request", Item(0) );
         }
         // or request to terminate?
         else if ( _request.asInteger() == 2 )
         {
            self->setProperty( "_request",  Item(0) );
            vm->retval( true );
            self->setProperty( "lastParsed", (int64) i );
            return;
         }
      }
   }

   self->setProperty( "lastParsed", (int64) i );
   vm->retval( true );
}

/*#
   @method expectValue CmdlineParser
   @brief Declares that current option needs a value.

   This method is to be called only from the onOption callback. When called,
   it suggests the parser that the received option requires a parameter,
   that should immediately follow.

   As the same option received by onOption() will be reported later on to onValue(),
   it is not necessary for the application to take note of the event. Simply, when
   receiving an option that needs a parameter, the application should call self.expectValue()
   and return.
*/

FALCON_FUNC  CmdlineParser_expectValue( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->setProperty( "_request", (int64) 1 );
}

/*#
   @method terminate CmdlineParser
   @brief Requests the parser to terminate parsing.

   This method should be called from inside one of the CmdlineParser
   class callbacks. Once called, the parser will immediately return true.
   The calling application can know the position of the last parsed parameter by
   accessing the lastParsed property, and handle the missing parameters as it prefers.
*/

FALCON_FUNC  CmdlineParser_terminate( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   self->setProperty( "_request", (int64) 2 );
}

FALCON_FUNC  CmdlineParser_usage( ::Falcon::VMachine *vm )
{
   vm->stdErr()->writeString( "The stub for \"CmdlineParser.usage()\" has been called.\n" );
   vm->stdErr()->writeString( "This class should be derived and the method usage() overloaded.\n" );
}

/*#
   @method onFree CmdlineParser
   @brief Called when the parser finds a free option.
   @param opt The free option being read.

   This callback method gets called by parse() when a command line parameter
   not being bound with any option is found. The overloaded method should
   check for this value respecting the host program command line semantic.
   In case the free option cannot be accepted, the method should either
   signal error and exit the application, ignore it or set an error indicator
   and request the parser to terminate by calling terminate().

   The parser won't signal error to the calling application, so, in case this free
   value cannot be accepted, an error state should be set in the application or
   in the parser object.
*/

/*#
   @method onOption CmdlineParser
   @brief Called when the parser finds an option.
   @param opt The option being read.

   This callback method gets called by parse() when an option is found. The
   overloaded method should check for the option being valid; in case it is not
   valid, it may either signal error and exit the application, ignore it or
   set an error indicator and request the parser to terminate by calling terminate().

   The parser won't signal error to the calling application, so, in case an invalid
   option is received, an error state should be set in the application or in the
   parser object.

   If the incoming option requires a parameter, this callback should call expectOption()
   before returning.
*/

/*#
   @method onSwitchOff CmdlineParser
   @brief Called when the parser finds a switch being turned off.
   @param opt The switch being turned off.

   This callback method gets called by parse() when a short option is
   immediately followed by a "-", indicating a "switch off" semantic.

   The overloaded method should check for the option being valid and being
   possibly target of a "switch off"; if not, it may either signal error and
   exit the application, ignore it or set an error indicator and request the
   parser to terminate by calling terminate().

   The parser won't signal error to the calling application, so, in case an invalid
   option is received, an error state should be set in the application or in the
   parser object.
*/

/*#
   @method onValue CmdlineParser
   @brief Called when the parser finds a value for a given option.
   @param opt The option being parsed.
   @param value The given value for that option.

   This callback method gets called by parse() when the parameter for a parametric
   option is read. The overloaded method should check for the option being valid;
   in case it is not valid, it may either signal error and exit the application,
   ignore it or set an error indicator and request the parser to terminate by calling
   terminate().

   The parser won't signal error to the calling application, so, in case an
   invalid option is received, an error state should be set in the application
   or in the parser object.
*/
}}

/* end of cmdlineparser.cpp */
