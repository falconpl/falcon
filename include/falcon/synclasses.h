/*
   FALCON - The Falcon Programming Language.
   FILE: synclasses.h

   Class holding all the Class reflector for syntactic tree elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 29 Dec 2011 12:07:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_SYNCLASSES_H
#define FALCON_SYNCLASSES_H

#include <falcon/setup.h>
#include <falcon/classes/classexpression.h>
#include <falcon/classes/classsyntree.h>
#include <falcon/classes/classstatement.h>

namespace Falcon {

class Engine;
class VMContext;
class GarbageToken;
class TreeStep;
class DataReader;
class DataWriter;
class VMContext;


/** Class holding all the Class reflector for syntactic tree elements.
 
 */
class SynClasses
{
public:
   /** Creates the syntactic classes.
    \param classSynTree the ClassSynTree instance held by the engine.
    \param classStatement the ClassStatement instance held by the engine.
    \param classExpr the ClassExpr instance held by the engine.
    
    */
   SynClasses( Class* classSynTree, Class* classStatement, Class* classExpr );      
   ~SynClasses();
   
   /** Subscribes all the syntactic classes to the engine.
    \param engine The engine where to subscribe the classes.
    
    This method adds all the Syntactic classes to the engine as registered
    classes.
    */
   void subscribe( Engine* engine );
   
   static GCToken* collect( const Class*, TreeStep*, int line );
   
   static void varExprInsert( VMContext* ctx, int pcount, TreeStep* step );   
   static void naryExprSet( VMContext* ctx, int pcount, TreeStep* step, int32 size );

   static inline void unaryExprSet( VMContext* ctx, int pcount, TreeStep* step ) {
      naryExprSet( ctx, pcount, step, 1 );
   }
   
   static inline void binaryExprSet( VMContext* ctx, int pcount, TreeStep* step ) {
      naryExprSet( ctx, pcount, step, 2 );
   }
   
   static inline void ternaryExprSet( VMContext* ctx, int pcount, TreeStep* step ) {
      naryExprSet( ctx, pcount, step, 3 );
   }
   
   static inline void zeroaryExprSet( VMContext*, int, TreeStep* ) {
      // no need to do anything
   }
   
   //======================================================================
   // Base classes
   //
   Class* m_cls_SynTree;
   Class* m_cls_Statement;
   Class* m_cls_Expression;
   Class* m_cls_TreeStep;
   
   #define FALCON_SYNCLASS_DECLARATOR_DECLARE
   #include <falcon/synclasses_list.h>      

   int m_dummy_end;
};

}

#define FALCON_DECLARE_SYN_CLASS( syntoken ) \
   static Class* syntoken = Engine::instance()->synclasses()->m_##syntoken; \
   m_handler = syntoken;

#endif	/* SYNCLASSES_H */

/* end of synclasses.h */
