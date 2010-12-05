/**
 *  \file gtk_VBox.cpp
 */

#include "gtk_VBox.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void VBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_VBox = mod->addClass( "GtkVBox", &VBox::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBox" ) );
    c_VBox->getClassDef()->addInheritance( in );

    c_VBox->getClassDef()->factory( &VBox::factory );
}


VBox::VBox( const Falcon::CoreClass* gen, const GtkVBox* box )
    :
    Gtk::CoreGObject( gen, (GObject*) box )
{}


Falcon::CoreObject* VBox::factory( const Falcon::CoreClass* gen, void* box, bool )
{
    return new VBox( gen, (GtkVBox*) box );
}


/*#
    @class GtkVBox
    @brief Vertical box class
    @optparam homogeneous (boolean, default true)
    @optparam spacing (integer, default 0)

    GtkVBox is a container that organizes child widgets into a single column.
 */
FALCON_FUNC VBox::init( VMARG )
{
    MYSELF;

    if ( self->getObject() )
        return;

    Item* i_homog = vm->param( 0 );
    Item* i_spacing = vm->param( 1 );

    gboolean homog = TRUE;
    gint spacing = 0;

    if ( i_homog )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_homog->isNil() || !i_homog->isBoolean() )
            throw_inv_params( "[B[,I]]" );
#endif
        homog = i_homog->asBoolean() ? TRUE : FALSE;
    }
    if ( i_spacing )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_spacing->isNil() || !i_spacing->isInteger() )
            throw_inv_params( "[B,[,I]]" );
#endif
        spacing = i_spacing->asInteger();
    }
    GtkWidget* gwdt = gtk_vbox_new( homog, spacing );
    self->setObject( (GObject*) gwdt );
}


} // Gtk
} // Falcon
