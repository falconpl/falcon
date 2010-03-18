/**
 *  \file gtk_HBox.cpp
 */

#include "gtk_HBox.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void HBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_HBox = mod->addClass( "HBox", &HBox::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Box" ) );
    c_HBox->getClassDef()->addInheritance( in );
}


/*#
    @class gtk.HBox
    @brief Horizontal box
    @optparam homogeneous (boolean, default true)
    @optparam spacing (integer, default 0)

    GtkHBox is a container that organizes child widgets into a single row.
 */
FALCON_FUNC HBox::init( VMARG )
{
    MYSELF;

    if ( self->getUserData() )
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
    Gtk::internal_add_slot( (GObject*) gwdt );
    self->setUserData( new GData( (GObject*) gwdt ) );
}


} // Gtk
} // Falcon
