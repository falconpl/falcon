/**
 *  \file gtk_Scale.cpp
 */

#include "gtk_Scale.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Scale::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Scale = mod->addClass( "GtkScale", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkRange" ) );
    c_Scale->getClassDef()->addInheritance( in );

    //c_Scale->getClassDef()->factory( &Scale::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_format_value",    &Scale::signal_format_value },
    { "set_digits",             &Scale::set_digits },
    { "set_draw_value",         &Scale::set_draw_value },
    { "set_value_pos",          &Scale::set_value_pos },
    { "get_digits",             &Scale::get_digits },
    { "get_draw_value",         &Scale::get_draw_value },
    { "get_value_pos",          &Scale::get_value_pos },
#if 0
    { "get_layout",             &Scale::get_layout },
    { "get_layout_offsets",     &Scale::get_layout_offsets },
#endif
    { "add_mark",               &Scale::add_mark },
    { "clear_marks",            &Scale::clear_marks },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Scale, meth->name, meth->cb );
}


Scale::Scale( const Falcon::CoreClass* gen, const GtkScale* range )
    :
    Gtk::CoreGObject( gen, (GObject*) range )
{}


Falcon::CoreObject* Scale::factory( const Falcon::CoreClass* gen, void* range, bool )
{
    return new Scale( gen, (GtkScale*) range );
}


/*#
    @class GtkScale
    @brief Base class for GtkHScale and GtkVScale

    A GtkScale is a slider control used to select a numeric value. To use it,
    you'll probably want to investigate the methods on its base class,
    GtkRange, in addition to the methods for GtkScale itself. To set the value of
    a scale, you would normally use gtk_range_set_value(). To detect changes to
    the value, you would normally use the "value_changed" signal.

    The GtkScale widget is an abstract class, used only for deriving the
    subclasses GtkHScale and GtkVScale. To create a scale widget, call
    gtk_hscale_new_with_range() or gtk_vscale_new_with_range().
 */


/*#
    @method signal_format_value GtkScale
    @brief Signal which allows you to change how the scale value is displayed.

    Connect a signal handler which returns a string representing value.
    That string will then be used to display the scale's value.
 */
FALCON_FUNC Scale::signal_format_value( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "format_value", (void*) &Scale::on_format_value, vm );
}


gchar* Scale::on_format_value( GtkScale* obj, gdouble value, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "format_value", false );

    if ( !cs || cs->empty() )
        return g_strdup_printf( "%0.*g", gtk_scale_get_digits( obj ), value );

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do // the first callback to return a string ends the loop
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_format_value", it ) )
            {
                printf(
                "[GtkScale::on_format_value] invalid callback (expected callable)\n" );
                return g_strdup_printf( "%0.*g", gtk_scale_get_digits( obj ), value );
            }
        }
        vm->pushParam( value );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isString() )
        {
            AutoCString cstr( it.asString() );
            return g_strdup( cstr.c_str() );
        }
        else
            iter.next();
    }
    while ( iter.hasCurrent() );

    printf( "[GtkScale::on_format_value] invalid callback (expected string)\n" );
    return g_strdup_printf( "%0.*g", gtk_scale_get_digits( obj ), value );
}


/*#
    @method set_digits GtkScale
    @brief Sets the number of decimal places that are displayed in the value.
    @param digits the number of decimal places to display, e.g. use 1 to display 1.0, 2 to display 1.00, etc

    Also causes the value of the adjustment to be rounded off to this number
    of digits, so the retrieved value matches the value the user saw.
 */
FALCON_FUNC Scale::set_digits( VMARG )
{
    Item* i_dig = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_dig || !i_dig->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_scale_set_digits( (GtkScale*)_obj, i_dig->asInteger() );
}


/*#
    @method set_draw_value GtkScale
    @brief Specifies whether the current value is displayed as a string next to the slider.
    @param TRUE to draw the value
 */
FALCON_FUNC Scale::set_draw_value( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isInteger() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_scale_set_draw_value( (GtkScale*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_value_pos GtkScale
    @brief Sets the position in which the current value is displayed.
    @param pos the position in which the current value is displayed (GtkPositionType).
 */
FALCON_FUNC Scale::set_value_pos( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_scale_set_value_pos( (GtkScale*)_obj, (GtkPositionType) i_pos->asInteger() );
}


/*#
    @method get_digits GtkScale
    @brief Gets the number of decimal places that are displayed in the value.
    @return the number of decimal places that are displayed
 */
FALCON_FUNC Scale::get_digits( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_scale_get_digits( (GtkScale*)_obj ) );
}


/*#
    @method get_draw_value GtkScale
    @brief Returns whether the current value is displayed as a string next to the slider.
    @return whether the current value is displayed as a string
 */
FALCON_FUNC Scale::get_draw_value( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_scale_get_draw_value( (GtkScale*)_obj ) );
}


/*#
    @method get_value_pos GtkScale
    @brief Gets the position in which the current value is displayed.
    @return the position in which the current value is displayed (GtkPositionType).
 */
FALCON_FUNC Scale::get_value_pos( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_scale_get_value_pos( (GtkScale*)_obj ) );
}


#if 0
FALCON_FUNC Scale::get_layout( VMARG );
FALCON_FUNC Scale::get_layout_offsets( VMARG );
#endif

/*#
    @method add_mark  GtkScale
	@brief Adds a mark at value. 
	@param value the value at which the mark is placed, must be between the lower and upper limits of the scales' adjustment
	@param position where to draw the mark. For a horizontal scale, GTK_POS_TOP is drawn above the scale, anything else below. For a vertical scale, GTK_POS_LEFT is drawn to the left of the scale, anything else to the right.
    
    @return nothing

	A mark is indicated visually by drawing a tick mark next to the scale, and GTK+ makes it easy for the user to position the scale exactly at the marks value. 
 */
FALCON_FUNC Scale::add_mark( VMARG )
{
    Item* i_pos = vm->param( 0 );
	Item* i_type = vm->param( 1 );

#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isOrdinal()  || 
		 !i_type || !i_type->isInteger()  )
        throw_inv_params( "N, <GtkPositionType>" );
#endif
	MYSELF;
    GET_OBJ( self );
	gtk_scale_add_mark( (GtkScale*)_obj, i_pos->forceNumeric(), (GtkPositionType)i_type->asInteger(), 0);

}


/*#
    @method clear_marks GtkScale
    @brief Removes any marks that have been added with gtk_scale_add_mark().
    @return nothing
 */
FALCON_FUNC Scale::clear_marks( VMARG )
{
	NO_ARGS
    MYSELF;
    GET_OBJ( self );
	gtk_scale_clear_marks( (GtkScale*)_obj ) ;	

}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
