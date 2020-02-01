/*
   FALCON - The Falcon Programming Language.
   FILE: cmdline/parser_ext.cpp

   The command line parser class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 06 Aug 2013 15:14:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The command line parser class
*/

/*#
   @beginmodule cmdline
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/function.h>
#include <falcon/textwriter.h>
#include <falcon/falconclass.h>
#include <falcon/falconinstance.h>
#include <falcon/class.h>
#include <falcon/vm.h>
#include <stdarg.h>

#include "cmdline_fm.h"


#define PARSE_LOCAL_LASTPARSED   0
#define PARSE_LOCAL_STAUS        1
   #define PARSE_LOCAL_STAUS_NONE       0
   #define PARSE_LOCAL_STAUS_WAITING    1
   #define PARSE_LOCAL_STAUS_ALLFREE    2
#define PARSE_LOCAL_CURRENTOPT   2
#define PARSE_LOCAL_ARGS         3

#define PARSE_PROPERTY_REQUEST      "_request"
#define PARSE_PROPERTY_LASTPARSED   "lastParsed"
#define PARSE_PROPERTY_ONVALUE      "onValue"
#define PARSE_PROPERTY_ONFREE       "onFree"
#define PARSE_PROPERTY_PASSMM       "passMinusMinus"
#define PARSE_PROPERTY_ONOPTION     "onOption"
#define PARSE_PROPERTY_ONSWITCHOFF  "onSwitchOff"

#define PARSE_PROPERTY_REQUEST_EXPECT  1
#define PARSE_PROPERTY_REQUEST_TERM    2


/*# @module cmdline Command line parser module
 @brief Provides facilities to parse complex command line parameters
 @ingroup feathers

 @beginmodule cmdline
 */

/*#
   @class Parser
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

   The Parser class, that is declared directly in the RTL module, provides a simple,
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
     accepts an arbitrary amount of free options, where the first N-1 options are
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
   in the Parser class. The parser will call the methods of the subclasses as
   it finds options in the argument vector; the callbacks will configure the application,
   report errors and mis-usage, terminate the program on fatal errors and communicate
   with the parser through member functions. For example, it is not necessary to declare
   in advance which are the parametric options; it's done on a per-option basis by calling
   the expectParam() method and returning to the parser.

   To use this feature, it is just necessary to declare a subclass of Parser and
   instance it, or derive an object from it, and call the parse() method.

   TheParser class is meant to be overloaded by scripts, so that the
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
   object MyParser from Parser

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


   @prop lastParsed An integer representing the last item parsed in the argument
          array before exit the parsing loop.
*/

namespace Falcon{
namespace {

class Parser: public FalconClass
{
   Parser(): FalconClass("Parser") {}
   virtual ~Parser() {}
};

/*#
   @method parse Parser
   @brief Starts command line parsing.
   @optparam args A specific string array that will be used as arguments to be parsed.
   @return true if the parsing is complete, false on error.

   Start the parsing process. If args parameter is not provided, the method gets
   the content of the @a args global vector defined in the Core module.

   Returns true if the parsing was complete, and false on error (for example,
   if some element in the array wasn't a string).
*/
FALCON_DECLARE_FUNCTION( parse, "args:[A]" )
FALCON_DEFINE_FUNCTION_P1( parse )
{
   ::Falcon::Feathers::ModuleCmdline* mod = static_cast< ::Falcon::Feathers::ModuleCmdline*>(this->module());

   FalconInstance *self = static_cast<FalconInstance*>(ctx->self().asInst());
   Item *i_params = ctx->param( 0 );

   if ( i_params == 0 )
   {
      // get the parameters from the VM args object
      i_params = ctx->resolveGlobal( "args", false );
      if ( i_params == 0 || ! i_params->isArray() ) {
         throw new CodeError( ErrorParam( e_undef_sym ).extra( "args" ).hard() );
      }
   }
   else if ( ! i_params->isArray() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).extra( "( A )" ) );
   }

   ItemArray *args = i_params->asArray();

   // zero request.
   self->setProperty( PARSE_PROPERTY_REQUEST, Item((int64) 0) );
   self->setProperty( PARSE_PROPERTY_LASTPARSED, Item((int64) 0) );


   // prepare for parameter parsing.
   ctx->addLocals(4);
   // character position
   ctx->local(PARSE_LOCAL_LASTPARSED)->setInteger(0);
   // Status
   ctx->local(PARSE_LOCAL_STAUS)->setInteger(PARSE_LOCAL_STAUS_NONE);
   // Last parsed option
   ctx->local(PARSE_LOCAL_CURRENTOPT)->setNil();
   // Parameters
   ctx->local(PARSE_LOCAL_ARGS)->setUser( args->handler(), args );

   // begin processing.
   ctx->stepIn(mod->m_stepGetOption);
}

/*#
   @method expectValue Parser
   @brief Declares that current optt_waitingValueion needs a value.

   This method is to be called only from the onOption callback. When called,
   it suggests the parser that the received option requires a parameter,
   that should immediately follow.

   As the same option received by onOption() will be reported later on to onValue(),
   it is not necessary for the application to take note of the event. Simply, when
   receiving an option that needs a parameter, the application should call self.expectValue()
   and return.
*/
FALCON_DECLARE_FUNCTION( expectValue, "" )
FALCON_DEFINE_FUNCTION_P1( expectValue )
{
   FalconInstance *self = static_cast<FalconInstance*>(ctx->self().asInst());
   self->setProperty( PARSE_PROPERTY_REQUEST, Item().setInteger(PARSE_PROPERTY_REQUEST_EXPECT) );
}

/*#
   @method terminate Parser
   @brief Requests the parser to terminate parsing.

   This method should be called from inside one of the Parser
   class callbacks. Once called, the parser will immediately return true.
   The calling application can know the position of the last parsed parameter by
   accessing the lastParsed property, and handle the missing parameters as it prefers.
*/
FALCON_DECLARE_FUNCTION( terminate, "" )
FALCON_DEFINE_FUNCTION_P1( terminate )
{
   FalconInstance *self = static_cast<FalconInstance*>(ctx->self().asInst());
   self->setProperty( PARSE_PROPERTY_REQUEST, Item().setInteger(PARSE_PROPERTY_REQUEST_TERM) );
}


FALCON_DECLARE_FUNCTION( usage, "" )
FALCON_DEFINE_FUNCTION_P1( usage )
{
   ctx->vm()->textOut()->write( "The stub for \"Parser.usage()\" has been called.\n" );
   ctx->vm()->textOut()->write( "This class should be derived and the method usage() overloaded.\n" );
}


/*#
   @method onFree Parser
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
   @method onOption Parser
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
   @method onSwitchOff Parser
   @brief Called when the parser finds a switch being turned off.
   @param opt The switch being turned off.

   This callback method gets called by parse() when a short option is
   immediately followed by a "-", indicating a "switch off" semantic.

   The overloaded method should check for the option being valid and being
   possibly target of a "switch off"; if not, it may either signal error and
   exit the application, ignore it or set an error indicator and request the
   parser to terminate by calling terminate().

   The parser won't signal error to the calling application, so, in case    // step called to get a value after an option
   PStep* m_stepGetValue;an invalid
   option is received, an error state should be set in the application or in the
   parser object.
*/

/*#
   @method onValue Parser
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
}

namespace Feathers {

//================================================================================================
// Main module part.
//================================================================================================

static void callACallback( VMContext* ctx, uint32 lastParsed, const String& method, int dummy,  ... )
{
   FalconInstance *self = static_cast<FalconInstance*>(ctx->self().asInst());

   // clear request prior the call
   self->setProperty( PARSE_PROPERTY_REQUEST, Item() );

   // get the required method
   Item i_method;
   self->getProperty( method, i_method );
   if ( i_method.isCallable() )
   {
     va_list params;
     va_start(params, dummy);

     ctx->pushData( i_method );
     uint32 count = 0;
     Item* p;
     while( (p = va_arg(params,Item*)) != 0 )
     {
        ctx->pushData( *p );
        ++count;
     }
     va_end(params);

     ctx->callInternal( i_method, count );
   }
   else
   {
     FalconInstance *self = static_cast<FalconInstance*>(ctx->self().asInst());
     self->setProperty( PARSE_PROPERTY_LASTPARSED, Item().setInteger(lastParsed) );
     ctx->returnFrame( Item().setBoolean(false) );
     return;
   }
}

static void saveStatus( VMContext* ctx, uint32 lastParsed, uint32 status, const Item& lastOption )
{
   ctx->local(PARSE_LOCAL_LASTPARSED)->setInteger(lastParsed);
   ctx->local(PARSE_LOCAL_STAUS)->setInteger(status);
   *ctx->local(PARSE_LOCAL_CURRENTOPT) = lastOption;
}


ModuleCmdline::ModuleCmdline():
         Module(FALCON_FEATHER_CMDLINE_NAME)
{
   class PStepGetOption: public PStep
   {
   public:
      PStepGetOption() { apply = &apply_; }
      virtual ~PStepGetOption() {}
      virtual void describeTo( String& target ) const
      {
         target = "ModuleCmdline::PStepGetOption";
      }

   private:

      static void apply_(const PStep*, VMContext* ctx )
      {
         // status.
         ModuleCmdline* mod = static_cast<ModuleCmdline*>(ctx->currentFrame().m_function->module());
         FalconInstance *self = static_cast<FalconInstance*>(ctx->self().asInst());
         uint32 i = ctx->local(PARSE_LOCAL_LASTPARSED)->asInteger();
         uint32 state = ctx->local(PARSE_LOCAL_STAUS)->asInteger();
         // copy by value as we may change the stack.
         Item currentOpt = *ctx->local(PARSE_LOCAL_CURRENTOPT);
         ItemArray* args = ctx->local(PARSE_LOCAL_ARGS)->asArray();

         // shall we pass minus-minus?
         Item i_passMM;
         self->getProperty( PARSE_PROPERTY_PASSMM, i_passMM );
         bool passMM = i_passMM.isTrue();

         for (; i < args->length(); i++ )
         {
              Item &i_opt = args->at( i );
              // ignore non-string elements in the array
              if ( !i_opt.isString() )
              {
                 continue;
              }

              String &opt = *i_opt.asString();
               // if we were expecting a value, we MUST consider ANYTHING as it was a value.
              if ( state == PARSE_LOCAL_STAUS_WAITING )
              {
                 saveStatus( ctx, i+1, 0, currentOpt );
                 callACallback( ctx, i, PARSE_PROPERTY_ONVALUE, 'e', &currentOpt, &i_opt, 0 );
                 return;
              }
              else if( opt.length() == 0 || (opt.getCharAt( 0 ) != '-' || opt.length() == 1) || state == PARSE_LOCAL_STAUS_ALLFREE )
              {
                 saveStatus( ctx, i+1, state, Item() );
                 callACallback( ctx, i, PARSE_PROPERTY_ONFREE, 'e', &i_opt, 0 );
                 return;
              }
              else if ( opt == "--" && ! passMM )
              {
                 state = PARSE_LOCAL_STAUS_ALLFREE;
                 continue; // to skip return value.
              }
              else {
                 // we have at least one '-', and length > 1
                 if ( opt.getCharAt( 1 ) == (uint32) '-' )
                 {
                    Item i_option = FALCON_GC_HANDLE( new String( opt == "--" ? "--" : opt.subString( 2 )) );

                    // we'll eventually change the status later, but now save the option.
                    saveStatus( ctx, i+1, PARSE_LOCAL_STAUS_NONE, i_option );
                    // get the module...
                    ModuleCmdline* mod = static_cast<ModuleCmdline*>( ctx->currentFrame().m_function->module() );
                    // ... see you after the call.
                    ctx->pushCode( mod->m_stepAfterCall );
                    callACallback( ctx, i, PARSE_PROPERTY_ONOPTION, 'e', &i_option, 0 );
                 }
                 else {
                    // are we a -x- switch-off setting?
                    if ( opt.length() == 3 && opt.getCharAt( 2 ) == (char_t) '-' )
                    {
                       // switch turnoff.
                       Item i_option = FALCON_GC_HANDLE( new String(opt.subString(1,2) ) );
                       saveStatus( ctx, i+1, PARSE_LOCAL_STAUS_NONE, Item() );
                       callACallback( ctx, i, PARSE_PROPERTY_ONSWITCHOFF, 'e', &i_option, 0 );
                    }
                    else {
                       Item i_option = FALCON_GC_HANDLE( new String(opt.subString(1) ) );
                       saveStatus( ctx, i+1, PARSE_LOCAL_STAUS_NONE, i_option );
                       ctx->pushCode(mod->m_stepAfterCall);
                       callACallback( ctx, i, PARSE_PROPERTY_ONOPTION, 'e', &i_option, 0 );
                    }
                 }

                 return;
              }
           }

           self->setProperty( PARSE_PROPERTY_LASTPARSED, Item().setInteger(i) );
           ctx->returnFrame( Item().setBoolean(true) );
      }
   };


   class PStepAfterCall: public PStep
   {
   public:
      PStepAfterCall() { apply = &apply_; }
      virtual ~PStepAfterCall() {}
      virtual void describeTo( String& target ) const
      {
         target = "ModParser::PStepAfterCall";
      }

   private:

      static void apply_(const PStep*, VMContext* ctx )
      {
         // we're not around anymore after this call.
         ctx->popCode();
         FalconInstance *self = static_cast<FalconInstance*>(ctx->self().asInst());

         Item i_request;
         self->getProperty( PARSE_PROPERTY_REQUEST, i_request );
         int32 req = (int32) i_request.asInteger();

         if( req == PARSE_PROPERTY_REQUEST_EXPECT )
         {
            ctx->local(PARSE_LOCAL_STAUS)->setInteger( PARSE_LOCAL_STAUS_WAITING );
         }
         else if( req == PARSE_PROPERTY_REQUEST_TERM )
         {
            ctx->returnFrame( Item().setBoolean(true) );
         }
         // else we have nothing peculiar to do.
      }
   };

   m_stepGetOption = new PStepGetOption;
   m_stepAfterCall = new PStepAfterCall;

   FalconClass* cls = new FalconClass("Parser");
   addMantra(cls);

   cls->addMethod( new Function_parse );
   cls->addMethod( new Function_terminate );
   cls->addMethod( new Function_usage );
   cls->addMethod( new Function_expectValue );

   cls->addProperty( PARSE_PROPERTY_REQUEST );
   cls->addProperty( PARSE_PROPERTY_LASTPARSED );
   cls->addProperty( PARSE_PROPERTY_PASSMM, Item().setBoolean(false) );

}

ModuleCmdline::~ModuleCmdline()
{
   delete m_stepGetOption;
   delete m_stepAfterCall;
}

}}

/* end of cmdline/parser.cpp */
