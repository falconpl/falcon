/*
   The Falcon Programming Language
   FILE: dynlib_mod.h

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   Internal logic functions - declarations.
*/

#ifndef dynlib_mod_H
#define dynlib_mod_H

#include <falcon/setup.h>
#include <falcon/falcondata.h>
#include <falcon/error.h>


namespace Falcon {
#define F_DYNLIB_MAX_PARAMS   32

// we need some old-type define to be sure to have one-byte specificators.
/** End of parameters */
#define F_DYNLIB_PTYPE_END   0
/** P */
#define F_DYNLIB_PTYPE_PTR    1
/** F */
#define F_DYNLIB_PTYPE_FLOAT  2
/** D */
#define F_DYNLIB_PTYPE_DOUBLE 3
/** I */
#define F_DYNLIB_PTYPE_I32    4
/** U */
#define F_DYNLIB_PTYPE_U32    6
/** L */
#define F_DYNLIB_PTYPE_LI     7
/** S */
#define F_DYNLIB_PTYPE_SZ     8
/** W */
#define F_DYNLIB_PTYPE_WZ     9
/** M */
#define F_DYNLIB_PTYPE_MB     10
/** Anything else */
#define F_DYNLIB_PTYPE_OPAQUE 11
/** ... */
#define F_DYNLIB_PTYPE_VAR    12
/** By pointer? */
#define F_DYNLIB_PTYPE_BYPTR  0x80


class FunctionAddress: public FalconData
{
   String m_name;
   byte *m_parsedParams;
   byte m_parsedReturn;

   /** Array of strings parallel to m_parsedParams where safety input types are stored. */
   String *m_safetyParams;

public:
   /**
      Function pointer
   */
   void *m_fAddress;

   /**
      The (original) parameter mask.
   */
   String m_paramMask;

   /**
      The return mask.
   */
   String m_returnMask;

   /** Should we guess our params? */
   bool m_bGuessParams;

   FunctionAddress( const String &name, void *address = 0 ):
      m_name( name ),
      m_parsedParams(0),
      m_parsedReturn( F_DYNLIB_PTYPE_PTR ),
      m_safetyParams(0),
      m_fAddress(address),
      m_bGuessParams( true )
   {}

   /**
      Copies an existing function pointer.
      /TODO
   */
   FunctionAddress( const FunctionAddress &other ):
      m_parsedParams(0),
      m_safetyParams(0),
      m_fAddress( other.m_fAddress ),
      m_paramMask( other.m_paramMask ),
      m_returnMask( other.m_returnMask ),
      m_bGuessParams( other.m_bGuessParams )
   {}

   // nothing needed to be done.
   virtual ~FunctionAddress();

   /** Overrides from falcon data.
      Actually, does nothing.
   */
   virtual void gcMark( VMachine *mp );

   /** Overrides from falcon data.
      Supports complete cloning of the underlying data.
   */
   virtual FalconData *clone() const;

   /* Calls the function address as required by the VM.
      In case the function has a parameter mask, the parameters
      stored in the VM are checked for correct type, and a
      parameter error is raised on mismatch.

      Otherwise, the parameters are turned into integers,
      strings or raw data pointers depending on their type.

      A parameter mismatch may crash the application.

      If a return mask is set for this FunctionPointer, the return
      value from the called function is stored as required in the
      VM return item, otherwise it is discarded.
      \param vm The VM from which parameters are taken and where return values are stored.
      \param firstPar the number of the first parameter in the parameter list that should be
          passed to the function (some parameters may have been used by the Falcon language
          call wrapper).
   */
   //void call( VMachine *vm, int32 firstPar = 0 ) const;

   /** Return this symbol's name */
   const String &name() const { return m_name; }

   /** Parse and eventually setup parameter masks.

      The parameter mask is parsed scanning an string containing tokens
      separated by whitespaces, ',' or ';' (they are the same).

      Each parameter specificator is either a single character or a "pseudoclass" name,
      which must be an arbitrary name long two characters more.

      The single character parameter specificator may be one of the following:

      - P - application specific opaque pointer (stored in a Falcon integer item).
      - F - 32 bit IEEE float format.
      - D - 64 bit IEEE double format.
      - I - 32 bit signed integer. This applies also to char and short int types, which
            are always padded to 32 bit integers when passed as parameters or returned.
      - U - 32 bit unsigned integer. This applies also to bytes and short unsigned int types, which
            are always padded to 32 bit integers when passed as parameters or returned.
      - L - 64 bit integers; as this is the maximum size allowed, sign is not relevant (the sign bit
            is always placed correctly both in parameter passing and return values).
      - S - UTF-8 encoded strings.
      - W - Wide character strings (compatible with UTF-16 in local byte ordering).
      - M - Memory buffers (MemBuf); this may also contain natively encoded strings.

      The special "..." token indicates that the function accepts a set of unknown
      parameters after that point, which will be treated as in unsafe mode.

      A pseudoclass name serves as a type safety constarint for the loaded library. Return
      values marked with pseudoclasses will generate a DynOpaque instance that will remember
      the class name and wrap the transported item. A pseudoclass parameter will check for the
      parameter passed by the Falcon script at the given position is of class DynOpaque and carrying
      the required pseudoclass type.

      Prepending a '$' sign in front of the parameter specificator will inform the parameter parsing
      system to pass the item by pointer to the underlying library.
      The Falcon script must pass the coresponding  parameter by reference, and after the underlying
      function returns, a possibly set-up or modified value is taken and stored into the parameter.
      Return specifiers cannot be prepended with '$'.

      \note Ideographic language users can define pseudoclasses with a single ideographic char,
      as a pseudoclass name is parsed if the count of characters in a substring is more than one,
      or if the only character is > 256U.
   */
   bool parseParams( const String &params );

   /**
      Parses the return value type specifier.

      On success, also configures this object's internal data.
   */
   bool parseReturn( const String &rval );

   /** Parses a single parameter, finding its type.
      \param mask A parameter list format.
      \param type gets the output type.
      \param begin First character in the mask to be parsed.
      \param end last character in the mask to be parsed.
      \return false if the parameter list is malformed, true (and type filled) on succesful parsing.
   */
   bool parseSingleParam( const String &mask, byte &type, uint32 begin=0, uint32 end=String::npos );

   /**
      Return the decoded nth param.
   */
   byte parsedParam( uint32 id ) const { return m_parsedParams[id]; }

   /** Return the nth pseudoclass name being present in the parsed parameters.
      If pid if out of range or the parameters are not parsed, the function will crash.
   */
   const String &pclassParam( uint32 pid ) const { return m_safetyParams[pid]; }
};


// our internal dynamic function class handler
class DynFuncManager: public Falcon::ObjectManager
{
public:
   // we derived it from falcon data
   virtual bool isFalconData() const;

   // our inner object doesn't need cache data.

   // it cannot create new data directly
   virtual void *onInit( Falcon::VMachine * );
   virtual void onDestroy( Falcon::VMachine *, void *user_data );
   virtual void *onClone( Falcon::VMachine *vm, void *user_data );
};


/**
 * Error for all DynLib errors.
 */
class DynLibError: public ::Falcon::Error
{
public:
   DynLibError():
      Error( "DynLibError" )
   {}

   DynLibError( const ErrorParam &params  ):
      Error( "DynLibError", params )
      {}
};

}

#endif

/* end of dynlib_mod.h */
