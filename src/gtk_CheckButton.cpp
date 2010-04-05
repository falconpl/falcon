/**
 *  \file gtk_CheckButton.cpp
 */

#include "gtk_CheckButton.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CheckButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CheckButton = mod->addClass( "GtkCheckButton", &CheckButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkToggleButton" ) );
    c_CheckButton->getClassDef()->addInheritance( in );

    c_CheckButton->getClassDef()->factory( &CheckButton::factory );
}


CheckButton::CheckButton( const Falcon::CoreClass* gen, const GtkCheckButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* CheckButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new CheckButton( gen, (GtkCheckButton*) btn );
}


/*#
    @class GtkCheckButton
    @brief Create widgets with a discrete toggle button
    @optparam label (string) the text for the check button.
    @optparam mnemonic (boolean, default false)

    A GtkCheckButton places a discrete gtk.ToggleButton next to a widget, (usually
    a GtkLabel). See the section on GtkToggleButton widgets for more information
    about toggle/check buttons.

    The important signal ('toggled') is also inherited from GtkToggleButton.
 */
FALCON_FUNC CheckButton::init( VMARG )
{
    MYSELF;

    if ( self->getGObject() )
        return;

    Item* i_lbl = vm->param( 0 );
    Item* i_mne = vm->param( 1 );
    GtkWidget* btn;

    if ( i_lbl )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_lbl->isNil() || i_lbl->isString() )
            throw_inv_params( "[S,B]" );
#endif
        AutoCString lbl( i_lbl->asString() );

        if ( i_mne )
        {
#ifndef NO_PARAMETER_CHECK
            if ( i_mne->isNil() || !i_mne->isBoolean() )
                throw_inv_params( "[S,B]" );
#endif
            if ( i_mne->asBoolean() )
                btn = gtk_check_button_new_with_mnemonic( lbl.c_str() );
            else
                btn = gtk_check_button_new_with_label( lbl.c_str() );
        }
        else
            btn = gtk_check_button_new_with_label( lbl.c_str() );
    }
    else
        btn = gtk_check_button_new();

    self->setGObject( (GObject*) btn );
}


} // Gtk
} // Falcon
