/*
   FALCON - The Falcon Programming Language.
   FILE: parser_proto.cpp

   Parser for Falcon source files -- prototype declarations handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Dec 2011 21:28:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_SYNCLASSES_ID_H
#define FALCON_SYNCLASSES_ID_H

/** ID For FOR classes. 
   ID used as Class::userFlag() safe in the TreeStep::cls() domain. 
 */
#define FALCON_SYNCLASS_ID_FORCLASSES 1

/** ID for the Rule class. 
   ID used as Class::userFlag() safe in the TreeStep::cls() domain. 
 */
#define FALCON_SYNCLASS_ID_RULE       2

/** ID for classes hosting else and elifs.
   ID used as Class::userFlag() safe in the TreeStep::cls() domain. 
 */
#define FALCON_SYNCLASS_ID_ELSEHOST   3

/** ID for classes hosting Case.
   ID used as Class::userFlag() safe in the TreeStep::cls() domain.
 */
#define FALCON_SYNCLASS_ID_CASEHOST   4

/** ID for classes hosting Autoexpression.
   ID used as Class::userFlag() safe in the TreeStep::cls() domain. 
 */
//#define FALCON_SYNCLASS_ID_AUTOEXPR   5

/** ID for classes hosting Catch and Finally clauses.
   ID used as Class::userFlag() safe in the TreeStep::cls() domain. 
 */
#define FALCON_SYNCLASS_ID_CATCHHOST  6

#define FALCON_SYNCLASS_ID_CALLFUNC   7

/** ID for classes hosting Case -- extended switch-like case. 
*/
#define FALCON_SYNCLASS_ID_SWITCH   8

/** ID for the Invoke expression.
*/
#define FALCON_SYNCLASS_ID_INVOKE   9

/** ID for the EPEX expression.
*/
#define FALCON_SYNCLASS_ID_EPEX     10

/** ID for the tree expression.
*/
#define FALCON_SYNCLASS_ID_TREE     11

/** ID for the symbol class
*/
#define FALCON_SYNCLASS_ID_SYMBOL   12

#endif	/* SYNCLASSES_ID_H */

/* end of synclasses_id.h */
