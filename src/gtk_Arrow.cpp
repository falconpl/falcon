/**
 *  \file gtk_Arrow.cpp
 */

#include "gtk_Arrow.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Arrow::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Arrow = mod->addClass( "Arrow", &Arrow::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Misc" ) );
    c_Arrow->getClassDef()->addInheritance( in );

    mod->addClassMethod( c_Arrow, "set", &Arrow::set );

}

/*#
    @class gtk.Arrow
    @brief Displays an arrow

    GtkArrow should be used to draw simple arrows that need to point in one of
    the four cardinal directions (up, down, left, or right). The style of the
    arrow can be one of shadow in, shadow out, etched in, or etched out.
    Note that these directions and style types may be ammended in versions of Gtk to come.

    GtkArrow will fill any space alloted to it, but since it is inherited from
    GtkMisc, it can be padded and/or aligned, to fill exactly the space the
    programmer desires.
 */

/*
    @init gtk.Arrow
    @brief Creates a new arrow
    @param arrow_type a valid GtkArrowType
    @param shadow_type a valid GtkShadowType
 */
FALCON_FUNC Arrow::init( VMARG )
{
    Item* i_type = vm->param( 0 );
    Item* i_shad = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_type || i_type->isNil() || !i_type->isInteger()
        || !i_shad || i_shad->isNil() || !i_shad->isInteger() )
        throw_inv_params( "arrow type, shadow type" );
#endif
    int type = i_type->asInteger();
    int shad = i_shad->asInteger();
#ifndef NO_PARAMETER_CHECK
    if ( type < 0 || type > 4
        || shad < 0 || shad > 4 )
        throw_inv_params( "out of bounds" );
#endif
    MYSELF;
    GtkWidget* wdt = gtk_arrow_new( (GtkArrowType) type, (GtkShadowType) shad );
    Gtk::internal_add_slot( (GObject*) wdt );
    self->setUserData( new GData( (GObject*) wdt ) );
}


/*#
    @method set gtk.Arrow
    @brief Sets the direction and style of the GtkArrow.
    @param arrow_type a valid GtkArrowType
    @param shadow_type a valid GtkShadowType
 */
FALCON_FUNC Arrow::set( VMARG )
{
    Item* i_type = vm->param( 0 );
    Item* i_shad = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_type || i_type->isNil() || !i_type->isInteger()
        || !i_shad || i_shad->isNil() || !i_shad->isInteger() )
        throw_inv_params( "arrow type, shadow type" );
#endif
    int type = i_type->asInteger();
    int shad = i_shad->asInteger();
#ifndef NO_PARAMETER_CHECK
    if ( type < 0 || type > 4
        || shad < 0 || shad > 4 )
        throw_inv_params( "out of bounds" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_arrow_set( (GtkArrow*)_obj, (GtkArrowType) type, (GtkShadowType) shad );
}


} // Gtk
} // Falcon
