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

class FunctionAddress: public FalconData
{
   String m_name;

public:
   /**
      Function pointer
   */
   void *m_fAddress;

   typedef enum {
      e_cdecl,
      e_stdcall
   } t_callProto;

   /**
      Call protocol
   */
   t_callProto m_callProto;

   const String m_paramMask;
   const String m_returnMask;

   bool m_bGuessParams;

   FunctionAddress( const String &name, void *address = 0 ):
      m_name( name ),
      m_fAddress(address),
      m_callProto( e_stdcall ),
      m_bGuessParams( false )
   {}

   /**
      Copies an existing function pointer.
   */
   FunctionAddress( const FunctionAddress &other ):
      m_fAddress( other.m_fAddress ),
      m_callProto( other.m_callProto ),
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

   /** Calls the function address as required by the VM.
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
   void call( VMachine *vm, int32 firstPar = 0 ) const;

   /** Return this symbol's name */
   const String &name() const { return m_name; }
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
