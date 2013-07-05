/*
   FALCON - The Falcon Programming Language
   FILE: stderrors.h

   Engine static/global data setup and initialization
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 15:30:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STDERRORS_H
#define _FALCON_STDERRORS_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/error.h>

namespace Falcon
{

   /** Class handler for AccessError exceptions.
    */
   FALCON_DECLARE_ERROR( AccessError )

   /** Class handler for AccessTypeError exceptions.
    */
   FALCON_DECLARE_ERROR( AccessTypeError )

   /** Class handler for CodeError exceptions.
    */
   FALCON_DECLARE_ERROR( CodeError )

   /** Class handler for GenericError exceptions.
    */
   FALCON_DECLARE_ERROR( GenericError )


   /** Class handler for InterruptedError exceptions.
    */
   FALCON_DECLARE_ERROR( InterruptedError )

   /** Class handler for IOError exceptions.
    */

   FALCON_DECLARE_ERROR( IOError )

   /** Class handler for LinkError exceptions.
    */
   FALCON_DECLARE_ERROR( LinkError )


   /** Class handler for OperandError exceptions.
    */
   FALCON_DECLARE_ERROR( OperandError )


   /** Class handler for UnsupportedError exceptions.
    */
   FALCON_DECLARE_ERROR( UnsupportedError )


   /** Class handler for EncodingError exceptions.
    */
   FALCON_DECLARE_ERROR( EncodingError )

   /** Class handler for ConcurrencyError exceptions.
    */
   FALCON_DECLARE_ERROR( ConcurrencyError )



   /** Class handler for SyntaxError exceptions.
    */
   FALCON_DECLARE_ERROR( SyntaxError )


   /** Class handler for ParamError exceptions.
    */
   FALCON_DECLARE_ERROR( ParamError )

   /** Class handler for MathError exceptions.
    */
   FALCON_DECLARE_ERROR( MathError )

   /** Class handler for MathError exceptions.
    */
   FALCON_DECLARE_ERROR( UnserializableError )

   /** Class handler for MathError exceptions.
    */
   FALCON_DECLARE_ERROR( TypeError )

}

#endif	/* _FALCON_STDERRORS_H */

/* end of stderrors.h */
