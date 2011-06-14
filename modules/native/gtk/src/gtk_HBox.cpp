/**
 *  \file gtk_HBox.cpp
 */

#include "gtk_HBox.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void HBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_HBox = mod->addClass( "GtkHBox", &HBox::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBox" ) );
    c_HBox->getClassDef()->addInheritance( in );

    c_HBox->getClassDef()->factory( &HBox::factory );
}


HBox::HBox( const Falcon::CoreClass* gen, const GtkHBox* box )
    :
    Gtk::CoreGObject( gen, (GObject*) box )
{}


Falcon::CoreObject* HBox::factory( const Falcon::CoreClass* gen, void* box, bool )
{
    return new HBox( gen, (GtkHBox*) box );
}


/*#
    @class GtkHBox
    @brief Horizontal box
    @optparam homogeneous (boolean, default true)
    @optparam spacing (integer, default 0)

    GtkHBox is a container that organizes child widgets into a single row.
 */
FALCON_FUNC HBox::init( VMARG )
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
    GtkWidget* gwdt = gtk_hbox_new( homog, spacing );

    self->setObject( (GObject*) gwdt );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
