/*
   FALCON - The Falcon Programming Language.
   FILE: errorclasses_list.h

   All the Class handling errors generated by the engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 15:30:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

// no single inclusion directive.

#ifdef FALCON_DEFINE_ERROR_CLASSES
   #undef FALCON_DECLARE_ERROR_CLASS
   #define FALCON_DECLARE_ERROR_CLASS( name ) \
   void* Class##name:: createInstance() const { return new name; }
   
#else

   #define FALCON_DECLARE_ERROR_CLASS( name ) \
      class Class##name: public ClassError\
      {\
      public:\
         inline Class##name(): ClassError( #name ) {} \
         inline virtual ~Class##name(){} \
         virtual void* createInstance() const;\
      };
#endif
/** Class handler for AccessError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( AccessError )

/** Class handler for AccessTypeError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( AccessTypeError )

/** Class handler for CodeError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( CodeError )

/** Class handler for GenericError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( GenericError )


/** Class handler for InterruptedError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( InterruptedError )

/** Class handler for IOError exceptions. 
 */

FALCON_DECLARE_ERROR_CLASS( IOError )

/** Class handler for LinkError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( LinkError )


/** Class handler for OperandError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( OperandError )


/** Class handler for UnsupportedError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( UnsupportedError )


/** Class handler for EncodingError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( EncodingError )


/** Class handler for SyntaxError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( SyntaxError )


/** Class handler for ParamError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( ParamError )

/** Class handler for MathError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( MathError )

/** Class handler for MathError exceptions. 
 */
FALCON_DECLARE_ERROR_CLASS( UnserializableError )
   
/* end of errorclasses_list.h */
   