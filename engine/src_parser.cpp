
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.4.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse         flc_src_parse
#define yylex           flc_src_lex
#define yyerror         flc_src_error
#define yylval          flc_src_lval
#define yychar          flc_src_char
#define yydebug         flc_src_debug
#define yynerrs         flc_src_nerrs


/* Copy the first part of user declarations.  */

/* Line 189 of yacc.c  */
#line 17 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  CTX_LINE  ( COMPILER->lexer()->contextStart() )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Line 189 of yacc.c  */
#line 124 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     UNB = 264,
     END = 265,
     DEF = 266,
     WHILE = 267,
     BREAK = 268,
     CONTINUE = 269,
     DROPPING = 270,
     IF = 271,
     ELSE = 272,
     ELIF = 273,
     FOR = 274,
     FORFIRST = 275,
     FORLAST = 276,
     FORMIDDLE = 277,
     SWITCH = 278,
     CASE = 279,
     DEFAULT = 280,
     SELECT = 281,
     SELF = 282,
     FSELF = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     INIT = 292,
     LOAD = 293,
     LAUNCH = 294,
     CONST_KW = 295,
     EXPORT = 296,
     IMPORT = 297,
     DIRECTIVE = 298,
     COLON = 299,
     FUNCDECL = 300,
     STATIC = 301,
     INNERFUNC = 302,
     FORDOT = 303,
     LISTPAR = 304,
     LOOP = 305,
     ENUM = 306,
     TRUE_TOKEN = 307,
     FALSE_TOKEN = 308,
     OUTER_STRING = 309,
     CLOSEPAR = 310,
     OPENPAR = 311,
     CLOSESQUARE = 312,
     OPENSQUARE = 313,
     DOT = 314,
     OPEN_GRAPH = 315,
     CLOSE_GRAPH = 316,
     ARROW = 317,
     VBAR = 318,
     ASSIGN_POW = 319,
     ASSIGN_SHL = 320,
     ASSIGN_SHR = 321,
     ASSIGN_BXOR = 322,
     ASSIGN_BOR = 323,
     ASSIGN_BAND = 324,
     ASSIGN_MOD = 325,
     ASSIGN_DIV = 326,
     ASSIGN_MUL = 327,
     ASSIGN_SUB = 328,
     ASSIGN_ADD = 329,
     OP_EQ = 330,
     OP_AS = 331,
     OP_TO = 332,
     COMMA = 333,
     QUESTION = 334,
     OR = 335,
     AND = 336,
     NOT = 337,
     LE = 338,
     GE = 339,
     LT = 340,
     GT = 341,
     NEQ = 342,
     EEQ = 343,
     OP_EXEQ = 344,
     PROVIDES = 345,
     OP_NOTIN = 346,
     OP_IN = 347,
     DIESIS = 348,
     ATSIGN = 349,
     CAP_CAP = 350,
     VBAR_VBAR = 351,
     AMPER_AMPER = 352,
     MINUS = 353,
     PLUS = 354,
     PERCENT = 355,
     SLASH = 356,
     STAR = 357,
     POW = 358,
     SHR = 359,
     SHL = 360,
     CAP_XOROOB = 361,
     CAP_ISOOB = 362,
     CAP_DEOOB = 363,
     CAP_OOB = 364,
     CAP_EVAL = 365,
     TILDE = 366,
     NEG = 367,
     AMPER = 368,
     DECREMENT = 369,
     INCREMENT = 370,
     DOLLAR = 371
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define UNB 264
#define END 265
#define DEF 266
#define WHILE 267
#define BREAK 268
#define CONTINUE 269
#define DROPPING 270
#define IF 271
#define ELSE 272
#define ELIF 273
#define FOR 274
#define FORFIRST 275
#define FORLAST 276
#define FORMIDDLE 277
#define SWITCH 278
#define CASE 279
#define DEFAULT 280
#define SELECT 281
#define SELF 282
#define FSELF 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define INIT 292
#define LOAD 293
#define LAUNCH 294
#define CONST_KW 295
#define EXPORT 296
#define IMPORT 297
#define DIRECTIVE 298
#define COLON 299
#define FUNCDECL 300
#define STATIC 301
#define INNERFUNC 302
#define FORDOT 303
#define LISTPAR 304
#define LOOP 305
#define ENUM 306
#define TRUE_TOKEN 307
#define FALSE_TOKEN 308
#define OUTER_STRING 309
#define CLOSEPAR 310
#define OPENPAR 311
#define CLOSESQUARE 312
#define OPENSQUARE 313
#define DOT 314
#define OPEN_GRAPH 315
#define CLOSE_GRAPH 316
#define ARROW 317
#define VBAR 318
#define ASSIGN_POW 319
#define ASSIGN_SHL 320
#define ASSIGN_SHR 321
#define ASSIGN_BXOR 322
#define ASSIGN_BOR 323
#define ASSIGN_BAND 324
#define ASSIGN_MOD 325
#define ASSIGN_DIV 326
#define ASSIGN_MUL 327
#define ASSIGN_SUB 328
#define ASSIGN_ADD 329
#define OP_EQ 330
#define OP_AS 331
#define OP_TO 332
#define COMMA 333
#define QUESTION 334
#define OR 335
#define AND 336
#define NOT 337
#define LE 338
#define GE 339
#define LT 340
#define GT 341
#define NEQ 342
#define EEQ 343
#define OP_EXEQ 344
#define PROVIDES 345
#define OP_NOTIN 346
#define OP_IN 347
#define DIESIS 348
#define ATSIGN 349
#define CAP_CAP 350
#define VBAR_VBAR 351
#define AMPER_AMPER 352
#define MINUS 353
#define PLUS 354
#define PERCENT 355
#define SLASH 356
#define STAR 357
#define POW 358
#define SHR 359
#define SHL 360
#define CAP_XOROOB 361
#define CAP_ISOOB 362
#define CAP_DEOOB 363
#define CAP_OOB 364
#define CAP_EVAL 365
#define TILDE 366
#define NEG 367
#define AMPER 368
#define DECREMENT 369
#define INCREMENT 370
#define DOLLAR 371




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union 
/* Line 214 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t
{

/* Line 214 of yacc.c  */
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"

   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
   Falcon::List *fal_genericList;



/* Line 214 of yacc.c  */
#line 412 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 424 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6743

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  117
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  165
/* YYNRULES -- Number of rules.  */
#define YYNRULES  468
/* YYNRULES -- Number of states.  */
#define YYNSTATES  852

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   371

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    61,    65,    69,    73,
      76,    79,    84,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   127,   131,   137,   141,   145,   146,   152,   155,   159,
     163,   167,   171,   172,   180,   184,   188,   189,   191,   192,
     199,   202,   206,   210,   214,   218,   219,   221,   222,   226,
     229,   233,   234,   239,   243,   247,   248,   251,   254,   258,
     261,   265,   269,   274,   279,   285,   289,   293,   294,   301,
     306,   311,   315,   316,   319,   322,   323,   326,   328,   330,
     332,   334,   338,   342,   346,   349,   353,   356,   360,   364,
     366,   367,   374,   378,   382,   383,   390,   394,   398,   399,
     406,   410,   414,   415,   422,   426,   430,   431,   434,   438,
     440,   441,   447,   448,   454,   455,   461,   462,   468,   469,
     470,   474,   475,   477,   480,   483,   486,   488,   492,   494,
     496,   498,   502,   504,   505,   512,   516,   520,   521,   524,
     528,   530,   531,   537,   538,   544,   545,   551,   552,   558,
     560,   564,   565,   567,   569,   573,   574,   581,   584,   588,
     589,   591,   593,   596,   599,   602,   607,   611,   617,   621,
     623,   627,   629,   631,   635,   639,   645,   648,   654,   655,
     663,   667,   673,   674,   681,   684,   685,   687,   691,   693,
     694,   695,   701,   702,   706,   709,   713,   716,   720,   724,
     728,   734,   740,   744,   747,   751,   755,   757,   761,   765,
     771,   777,   785,   793,   801,   809,   814,   819,   824,   829,
     836,   843,   847,   852,   857,   859,   863,   867,   871,   873,
     877,   881,   885,   889,   890,   898,   902,   905,   906,   910,
     911,   917,   918,   921,   923,   927,   930,   931,   934,   938,
     939,   942,   944,   946,   948,   950,   952,   954,   955,   963,
     969,   974,   979,   984,   989,   990,   993,   995,   997,   998,
    1006,  1007,  1010,  1012,  1017,  1019,  1022,  1024,  1026,  1027,
    1035,  1038,  1041,  1042,  1045,  1047,  1049,  1051,  1053,  1055,
    1056,  1061,  1063,  1065,  1068,  1072,  1076,  1078,  1081,  1085,
    1089,  1091,  1093,  1095,  1097,  1099,  1101,  1103,  1105,  1107,
    1109,  1111,  1113,  1115,  1117,  1119,  1121,  1123,  1125,  1126,
    1128,  1130,  1132,  1135,  1138,  1141,  1145,  1149,  1153,  1156,
    1160,  1165,  1170,  1175,  1180,  1185,  1190,  1195,  1200,  1205,
    1210,  1215,  1218,  1222,  1225,  1228,  1231,  1234,  1238,  1242,
    1246,  1250,  1254,  1258,  1262,  1266,  1269,  1273,  1277,  1281,
    1284,  1287,  1290,  1293,  1296,  1299,  1302,  1305,  1308,  1310,
    1312,  1314,  1316,  1318,  1320,  1323,  1325,  1330,  1336,  1340,
    1342,  1344,  1348,  1354,  1358,  1362,  1366,  1370,  1374,  1378,
    1382,  1386,  1390,  1394,  1398,  1402,  1406,  1411,  1416,  1422,
    1430,  1435,  1439,  1440,  1447,  1448,  1455,  1456,  1463,  1468,
    1472,  1475,  1478,  1481,  1484,  1485,  1492,  1498,  1504,  1509,
    1513,  1516,  1520,  1524,  1527,  1531,  1535,  1539,  1543,  1548,
    1550,  1554,  1556,  1560,  1561,  1563,  1565,  1569,  1573
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     118,     0,    -1,   119,    -1,    -1,   119,   120,    -1,   121,
      -1,    10,     3,    -1,    24,     1,     3,    -1,   123,    -1,
     220,    -1,   200,    -1,   223,    -1,   246,    -1,   241,    -1,
     124,    -1,   214,    -1,   215,    -1,   217,    -1,     4,    -1,
      98,     4,    -1,    38,     6,     3,    -1,    38,     7,     3,
      -1,    38,     1,     3,    -1,   125,    -1,   218,    -1,     3,
      -1,    45,     1,     3,    -1,    34,     1,     3,    -1,    32,
       1,     3,    -1,     1,     3,    -1,   261,     3,    -1,   277,
      75,   261,     3,    -1,   277,    75,   261,    78,   277,     3,
      -1,   127,    -1,   128,    -1,   132,    -1,   149,    -1,   164,
      -1,   179,    -1,   135,    -1,   146,    -1,   147,    -1,   190,
      -1,   199,    -1,   255,    -1,   251,    -1,   213,    -1,   155,
      -1,   156,    -1,   157,    -1,   258,    -1,   258,    75,   261,
      -1,   126,    78,   258,    75,   261,    -1,    11,   126,     3,
      -1,    11,     1,     3,    -1,    -1,   130,   129,   145,    10,
       3,    -1,   131,   124,    -1,    12,   261,     3,    -1,    12,
       1,     3,    -1,    12,   261,    44,    -1,    12,     1,    44,
      -1,    -1,    50,     3,   133,   145,    10,   134,     3,    -1,
      50,    44,   124,    -1,    50,     1,     3,    -1,    -1,   261,
      -1,    -1,   137,   136,   145,   139,    10,     3,    -1,   138,
     124,    -1,    16,   261,     3,    -1,    16,     1,     3,    -1,
      16,   261,    44,    -1,    16,     1,    44,    -1,    -1,   142,
      -1,    -1,   141,   140,   145,    -1,    17,     3,    -1,    17,
       1,     3,    -1,    -1,   144,   143,   145,   139,    -1,    18,
     261,     3,    -1,    18,     1,     3,    -1,    -1,   145,   124,
      -1,    13,     3,    -1,    13,     1,     3,    -1,    14,     3,
      -1,    14,    15,     3,    -1,    14,     1,     3,    -1,    19,
     280,    92,   261,    -1,    19,   258,    75,   151,    -1,    19,
     280,    92,     1,     3,    -1,    19,     1,     3,    -1,   148,
      44,   124,    -1,    -1,   148,     3,   150,   153,    10,     3,
      -1,   261,    77,   261,   152,    -1,   261,    77,   261,     1,
      -1,   261,    77,     1,    -1,    -1,    78,   261,    -1,    78,
       1,    -1,    -1,   154,   153,    -1,   124,    -1,   158,    -1,
     160,    -1,   162,    -1,    48,   261,     3,    -1,    48,     1,
       3,    -1,   104,   277,     3,    -1,   104,     3,    -1,    86,
     277,     3,    -1,    86,     3,    -1,   104,     1,     3,    -1,
      86,     1,     3,    -1,    54,    -1,    -1,    20,     3,   159,
     145,    10,     3,    -1,    20,    44,   124,    -1,    20,     1,
       3,    -1,    -1,    21,     3,   161,   145,    10,     3,    -1,
      21,    44,   124,    -1,    21,     1,     3,    -1,    -1,    22,
       3,   163,   145,    10,     3,    -1,    22,    44,   124,    -1,
      22,     1,     3,    -1,    -1,   166,   165,   167,   173,    10,
       3,    -1,    23,   261,     3,    -1,    23,     1,     3,    -1,
      -1,   167,   168,    -1,   167,     1,     3,    -1,     3,    -1,
      -1,    24,   177,     3,   169,   145,    -1,    -1,    24,   177,
      44,   170,   124,    -1,    -1,    24,     1,     3,   171,   145,
      -1,    -1,    24,     1,    44,   172,   124,    -1,    -1,    -1,
     175,   174,   176,    -1,    -1,    25,    -1,    25,     1,    -1,
       3,   145,    -1,    44,   124,    -1,   178,    -1,   177,    78,
     178,    -1,     8,    -1,   122,    -1,     7,    -1,   122,    77,
     122,    -1,     6,    -1,    -1,   181,   180,   182,   173,    10,
       3,    -1,    26,   261,     3,    -1,    26,     1,     3,    -1,
      -1,   182,   183,    -1,   182,     1,     3,    -1,     3,    -1,
      -1,    24,   188,     3,   184,   145,    -1,    -1,    24,   188,
      44,   185,   124,    -1,    -1,    24,     1,     3,   186,   145,
      -1,    -1,    24,     1,    44,   187,   124,    -1,   189,    -1,
     188,    78,   189,    -1,    -1,     4,    -1,     6,    -1,    29,
      44,   124,    -1,    -1,   192,   191,   145,   193,    10,     3,
      -1,    29,     3,    -1,    29,     1,     3,    -1,    -1,   194,
      -1,   195,    -1,   194,   195,    -1,   196,   145,    -1,    30,
       3,    -1,    30,    92,   258,     3,    -1,    30,   197,     3,
      -1,    30,   197,    92,   258,     3,    -1,    30,     1,     3,
      -1,   198,    -1,   197,    78,   198,    -1,     4,    -1,     6,
      -1,    31,   261,     3,    -1,    31,     1,     3,    -1,   201,
     208,   145,    10,     3,    -1,   203,   124,    -1,   205,    56,
     206,    55,     3,    -1,    -1,   205,    56,   206,     1,   202,
      55,     3,    -1,   205,     1,     3,    -1,   205,    56,   206,
      55,    44,    -1,    -1,   205,    56,     1,   204,    55,    44,
      -1,    45,     6,    -1,    -1,   207,    -1,   206,    78,   207,
      -1,     6,    -1,    -1,    -1,   211,   209,   145,    10,     3,
      -1,    -1,   212,   210,   124,    -1,    46,     3,    -1,    46,
       1,     3,    -1,    46,    44,    -1,    46,     1,    44,    -1,
      39,   263,     3,    -1,    39,     1,     3,    -1,    40,     6,
      75,   257,     3,    -1,    40,     6,    75,     1,     3,    -1,
      40,     1,     3,    -1,    41,     3,    -1,    41,   216,     3,
      -1,    41,     1,     3,    -1,     6,    -1,   216,    78,     6,
      -1,    42,   219,     3,    -1,    42,   219,    33,     6,     3,
      -1,    42,   219,    33,     7,     3,    -1,    42,   219,    33,
       6,    76,     6,     3,    -1,    42,   219,    33,     7,    76,
       6,     3,    -1,    42,   219,    33,     6,    92,     6,     3,
      -1,    42,   219,    33,     7,    92,     6,     3,    -1,    42,
       6,     1,     3,    -1,    42,   219,     1,     3,    -1,    42,
      33,     6,     3,    -1,    42,    33,     7,     3,    -1,    42,
      33,     6,    76,     6,     3,    -1,    42,    33,     7,    76,
       6,     3,    -1,    42,     1,     3,    -1,     6,    44,   257,
       3,    -1,     6,    44,     1,     3,    -1,     6,    -1,   219,
      78,     6,    -1,    43,   221,     3,    -1,    43,     1,     3,
      -1,   222,    -1,   221,    78,   222,    -1,     6,    75,     6,
      -1,     6,    75,     7,    -1,     6,    75,   122,    -1,    -1,
      32,     6,   224,   225,   232,    10,     3,    -1,   226,   228,
       3,    -1,     1,     3,    -1,    -1,    56,   206,    55,    -1,
      -1,    56,   206,     1,   227,    55,    -1,    -1,    33,   229,
      -1,   230,    -1,   229,    78,   230,    -1,     6,   231,    -1,
      -1,    56,    55,    -1,    56,   277,    55,    -1,    -1,   232,
     233,    -1,     3,    -1,   200,    -1,   236,    -1,   237,    -1,
     234,    -1,   218,    -1,    -1,    37,     3,   235,   208,   145,
      10,     3,    -1,    46,     6,    75,   257,     3,    -1,     6,
      75,   261,     3,    -1,   238,   239,    10,     3,    -1,    58,
       6,    57,     3,    -1,    58,    37,    57,     3,    -1,    -1,
     239,   240,    -1,     3,    -1,   200,    -1,    -1,    51,     6,
     242,     3,   243,    10,     3,    -1,    -1,   243,   244,    -1,
       3,    -1,     6,    75,   257,   245,    -1,   218,    -1,     6,
     245,    -1,     3,    -1,    78,    -1,    -1,    34,     6,   247,
     248,   249,    10,     3,    -1,   228,     3,    -1,     1,     3,
      -1,    -1,   249,   250,    -1,     3,    -1,   200,    -1,   236,
      -1,   234,    -1,   218,    -1,    -1,    36,   252,   253,     3,
      -1,   254,    -1,     1,    -1,   254,     1,    -1,   253,    78,
     254,    -1,   253,    78,     1,    -1,     6,    -1,    35,     3,
      -1,    35,   261,     3,    -1,    35,     1,     3,    -1,     8,
      -1,     9,    -1,    52,    -1,    53,    -1,     4,    -1,     5,
      -1,     7,    -1,     8,    -1,     9,    -1,    52,    -1,    53,
      -1,   122,    -1,     5,    -1,     7,    -1,     6,    -1,   258,
      -1,    27,    -1,    28,    -1,    -1,     3,    -1,   256,    -1,
     259,    -1,   113,     6,    -1,   113,     4,    -1,   113,    27,
      -1,   113,    59,     6,    -1,   113,    59,     4,    -1,   113,
      59,    27,    -1,    98,   261,    -1,     6,    63,   261,    -1,
     261,    99,   260,   261,    -1,   261,    98,   260,   261,    -1,
     261,   102,   260,   261,    -1,   261,   101,   260,   261,    -1,
     261,   100,   260,   261,    -1,   261,   103,   260,   261,    -1,
     261,    97,   260,   261,    -1,   261,    96,   260,   261,    -1,
     261,    95,   260,   261,    -1,   261,   105,   260,   261,    -1,
     261,   104,   260,   261,    -1,   111,   261,    -1,   261,    87,
     261,    -1,   261,   115,    -1,   115,   261,    -1,   261,   114,
      -1,   114,   261,    -1,   261,    88,   261,    -1,   261,    89,
     261,    -1,   261,    86,   261,    -1,   261,    85,   261,    -1,
     261,    84,   261,    -1,   261,    83,   261,    -1,   261,    81,
     261,    -1,   261,    80,   261,    -1,    82,   261,    -1,   261,
      92,   261,    -1,   261,    91,   261,    -1,   261,    90,     6,
      -1,   116,   258,    -1,   116,     4,    -1,    94,   261,    -1,
      93,   261,    -1,   110,   261,    -1,   109,   261,    -1,   108,
     261,    -1,   107,   261,    -1,   106,   261,    -1,   265,    -1,
     267,    -1,   271,    -1,   263,    -1,   273,    -1,   275,    -1,
     261,   262,    -1,   274,    -1,   261,    58,   261,    57,    -1,
     261,    58,   102,   261,    57,    -1,   261,    59,     6,    -1,
     276,    -1,   262,    -1,   261,    75,   261,    -1,   261,    75,
     261,    78,   277,    -1,   261,    74,   261,    -1,   261,    73,
     261,    -1,   261,    72,   261,    -1,   261,    71,   261,    -1,
     261,    70,   261,    -1,   261,    64,   261,    -1,   261,    69,
     261,    -1,   261,    68,   261,    -1,   261,    67,   261,    -1,
     261,    65,   261,    -1,   261,    66,   261,    -1,    56,   261,
      55,    -1,    58,    44,    57,    -1,    58,   261,    44,    57,
      -1,    58,    44,   261,    57,    -1,    58,   261,    44,   261,
      57,    -1,    58,   261,    44,   261,    44,   261,    57,    -1,
     261,    56,   277,    55,    -1,   261,    56,    55,    -1,    -1,
     261,    56,   277,     1,   264,    55,    -1,    -1,    45,   266,
     269,   208,   145,    10,    -1,    -1,    60,   268,   270,   208,
     145,    61,    -1,    56,   206,    55,     3,    -1,    56,   206,
       1,    -1,     1,     3,    -1,   206,    62,    -1,   206,     1,
      -1,     1,    62,    -1,    -1,    47,   272,   269,   208,   145,
      10,    -1,   261,    79,   261,    44,   261,    -1,   261,    79,
     261,    44,     1,    -1,   261,    79,   261,     1,    -1,   261,
      79,     1,    -1,    58,    57,    -1,    58,   277,    57,    -1,
      58,   277,     1,    -1,    49,    57,    -1,    49,   278,    57,
      -1,    49,   278,     1,    -1,    58,    62,    57,    -1,    58,
     281,    57,    -1,    58,   281,     1,    57,    -1,   261,    -1,
     277,    78,   261,    -1,   261,    -1,   278,   279,   261,    -1,
      -1,    78,    -1,   258,    -1,   280,    78,   258,    -1,   261,
      62,   261,    -1,   281,    78,   261,    62,   261,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   197,   197,   200,   202,   206,   207,   208,   212,   213,
     214,   219,   224,   229,   234,   239,   240,   241,   245,   246,
     250,   256,   262,   269,   270,   271,   272,   273,   274,   275,
     280,   291,   297,   311,   312,   313,   314,   315,   316,   317,
     318,   319,   320,   321,   322,   323,   324,   325,   326,   327,
     331,   334,   340,   348,   350,   355,   355,   369,   377,   378,
     382,   383,   387,   387,   402,   408,   415,   416,   420,   420,
     435,   445,   446,   450,   451,   455,   457,   458,   458,   467,
     468,   473,   473,   485,   486,   489,   491,   497,   506,   514,
     524,   533,   541,   546,   554,   559,   568,   587,   586,   607,
     613,   617,   624,   625,   626,   629,   631,   635,   642,   643,
     644,   648,   661,   669,   673,   679,   685,   692,   697,   706,
     716,   716,   730,   739,   743,   743,   756,   765,   769,   769,
     785,   794,   798,   798,   815,   816,   823,   825,   826,   830,
     832,   831,   842,   842,   854,   854,   866,   866,   882,   885,
     884,   897,   898,   899,   902,   903,   909,   910,   914,   923,
     935,   946,   957,   978,   978,   995,   996,  1003,  1005,  1006,
    1010,  1012,  1011,  1022,  1022,  1035,  1035,  1047,  1047,  1065,
    1066,  1069,  1070,  1082,  1103,  1110,  1109,  1128,  1129,  1132,
    1134,  1138,  1139,  1143,  1148,  1166,  1186,  1196,  1207,  1215,
    1216,  1220,  1232,  1255,  1256,  1263,  1273,  1282,  1283,  1283,
    1287,  1291,  1292,  1292,  1299,  1399,  1401,  1402,  1406,  1421,
    1424,  1423,  1435,  1434,  1449,  1450,  1454,  1455,  1464,  1468,
    1476,  1486,  1491,  1503,  1512,  1519,  1527,  1532,  1540,  1545,
    1550,  1555,  1575,  1594,  1599,  1604,  1609,  1623,  1628,  1633,
    1638,  1643,  1652,  1657,  1664,  1670,  1682,  1687,  1695,  1696,
    1700,  1704,  1708,  1722,  1721,  1784,  1787,  1793,  1795,  1796,
    1796,  1802,  1804,  1808,  1809,  1813,  1837,  1838,  1839,  1846,
    1848,  1852,  1853,  1856,  1875,  1895,  1896,  1900,  1900,  1934,
    1956,  1984,  1996,  2004,  2017,  2019,  2024,  2025,  2037,  2036,
    2080,  2082,  2086,  2087,  2091,  2092,  2099,  2099,  2108,  2107,
    2174,  2175,  2181,  2183,  2187,  2188,  2191,  2210,  2211,  2220,
    2219,  2237,  2238,  2243,  2248,  2249,  2256,  2272,  2273,  2274,
    2282,  2283,  2284,  2285,  2286,  2287,  2288,  2292,  2293,  2294,
    2295,  2296,  2297,  2298,  2302,  2320,  2321,  2322,  2342,  2344,
    2348,  2349,  2350,  2351,  2352,  2353,  2354,  2355,  2356,  2357,
    2358,  2384,  2385,  2405,  2429,  2446,  2447,  2448,  2449,  2450,
    2451,  2452,  2453,  2454,  2455,  2456,  2457,  2458,  2459,  2460,
    2461,  2462,  2463,  2464,  2465,  2466,  2467,  2468,  2469,  2470,
    2471,  2472,  2473,  2474,  2475,  2476,  2477,  2478,  2479,  2480,
    2481,  2482,  2483,  2484,  2486,  2491,  2495,  2500,  2506,  2515,
    2516,  2518,  2523,  2530,  2531,  2532,  2533,  2534,  2535,  2536,
    2537,  2538,  2539,  2540,  2541,  2546,  2549,  2552,  2555,  2558,
    2564,  2570,  2575,  2575,  2585,  2584,  2627,  2626,  2678,  2679,
    2683,  2690,  2691,  2695,  2703,  2702,  2751,  2756,  2763,  2770,
    2780,  2781,  2785,  2793,  2794,  2798,  2807,  2808,  2809,  2817,
    2818,  2822,  2823,  2826,  2827,  2830,  2836,  2843,  2844
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "UNB", "END", "DEF", "WHILE", "BREAK", "CONTINUE",
  "DROPPING", "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST",
  "FORMIDDLE", "SWITCH", "CASE", "DEFAULT", "SELECT", "SELF", "FSELF",
  "TRY", "CATCH", "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL",
  "INIT", "LOAD", "LAUNCH", "CONST_KW", "EXPORT", "IMPORT", "DIRECTIVE",
  "COLON", "FUNCDECL", "STATIC", "INNERFUNC", "FORDOT", "LISTPAR", "LOOP",
  "ENUM", "TRUE_TOKEN", "FALSE_TOKEN", "OUTER_STRING", "CLOSEPAR",
  "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "OPEN_GRAPH",
  "CLOSE_GRAPH", "ARROW", "VBAR", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR",
  "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV",
  "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_AS", "OP_TO",
  "COMMA", "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ",
  "EEQ", "OP_EXEQ", "PROVIDES", "OP_NOTIN", "OP_IN", "DIESIS", "ATSIGN",
  "CAP_CAP", "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS", "PERCENT",
  "SLASH", "STAR", "POW", "SHR", "SHL", "CAP_XOROOB", "CAP_ISOOB",
  "CAP_DEOOB", "CAP_OOB", "CAP_EVAL", "TILDE", "NEG", "AMPER", "DECREMENT",
  "INCREMENT", "DOLLAR", "$accept", "input", "body", "line",
  "toplevel_statement", "INTNUM_WITH_MINUS", "load_statement", "statement",
  "base_statement", "assignment_def_list", "def_statement",
  "while_statement", "$@1", "while_decl", "while_short_decl",
  "loop_statement", "$@2", "loop_terminator", "if_statement", "$@3",
  "if_decl", "if_short_decl", "elif_or_else", "$@4", "else_decl",
  "elif_statement", "$@5", "elif_decl", "statement_list",
  "break_statement", "continue_statement", "forin_header",
  "forin_statement", "$@6", "for_to_expr", "for_to_step_clause",
  "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "outer_print_statement", "first_loop_block",
  "$@7", "last_loop_block", "$@8", "middle_loop_block", "$@9",
  "switch_statement", "$@10", "switch_decl", "case_list", "case_statement",
  "$@11", "$@12", "$@13", "$@14", "default_statement", "$@15",
  "default_decl", "default_body", "case_expression_list", "case_element",
  "select_statement", "$@16", "select_decl", "selcase_list",
  "selcase_statement", "$@17", "$@18", "$@19", "$@20",
  "selcase_expression_list", "selcase_element", "try_statement", "$@21",
  "try_decl", "catch_statements", "catch_list", "catch_statement",
  "catch_decl", "catchcase_element_list", "catchcase_element",
  "raise_statement", "func_statement", "func_decl", "$@22",
  "func_decl_short", "$@23", "func_begin", "param_list", "param_symbol",
  "static_block", "$@24", "$@25", "static_decl", "static_short_decl",
  "launch_statement", "const_statement", "export_statement",
  "export_symbol_list", "import_statement", "attribute_statement",
  "import_symbol_list", "directive_statement", "directive_pair_list",
  "directive_pair", "class_decl", "$@26", "class_def_inner",
  "class_param_list", "$@27", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "class_statement_list",
  "class_statement", "init_decl", "$@28", "property_decl", "state_decl",
  "state_heading", "state_statement_list", "state_statement",
  "enum_statement", "$@29", "enum_statement_list", "enum_item_decl",
  "enum_item_terminator", "object_decl", "$@30", "object_decl_inner",
  "object_statement_list", "object_statement", "global_statement", "$@31",
  "global_symbol_list", "globalized_symbol", "return_statement",
  "const_atom_non_minus", "const_atom", "atomic_symbol", "var_atom",
  "OPT_EOL", "expression", "range_decl", "func_call", "$@32",
  "nameless_func", "$@33", "nameless_block", "$@34",
  "nameless_func_decl_inner", "nameless_block_decl_inner", "innerfunc",
  "$@35", "iif_expr", "array_decl", "dotarray_decl", "dict_decl",
  "expression_list", "listpar_expression_list", "listpar_comma",
  "symbol_list", "expression_pair_list", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   117,   118,   119,   119,   120,   120,   120,   121,   121,
     121,   121,   121,   121,   121,   121,   121,   121,   122,   122,
     123,   123,   123,   124,   124,   124,   124,   124,   124,   124,
     125,   125,   125,   125,   125,   125,   125,   125,   125,   125,
     125,   125,   125,   125,   125,   125,   125,   125,   125,   125,
     126,   126,   126,   127,   127,   129,   128,   128,   130,   130,
     131,   131,   133,   132,   132,   132,   134,   134,   136,   135,
     135,   137,   137,   138,   138,   139,   139,   140,   139,   141,
     141,   143,   142,   144,   144,   145,   145,   146,   146,   147,
     147,   147,   148,   148,   148,   148,   149,   150,   149,   151,
     151,   151,   152,   152,   152,   153,   153,   154,   154,   154,
     154,   155,   155,   156,   156,   156,   156,   156,   156,   157,
     159,   158,   158,   158,   161,   160,   160,   160,   163,   162,
     162,   162,   165,   164,   166,   166,   167,   167,   167,   168,
     169,   168,   170,   168,   171,   168,   172,   168,   173,   174,
     173,   175,   175,   175,   176,   176,   177,   177,   178,   178,
     178,   178,   178,   180,   179,   181,   181,   182,   182,   182,
     183,   184,   183,   185,   183,   186,   183,   187,   183,   188,
     188,   189,   189,   189,   190,   191,   190,   192,   192,   193,
     193,   194,   194,   195,   196,   196,   196,   196,   196,   197,
     197,   198,   198,   199,   199,   200,   200,   201,   202,   201,
     201,   203,   204,   203,   205,   206,   206,   206,   207,   208,
     209,   208,   210,   208,   211,   211,   212,   212,   213,   213,
     214,   214,   214,   215,   215,   215,   216,   216,   217,   217,
     217,   217,   217,   217,   217,   217,   217,   217,   217,   217,
     217,   217,   218,   218,   219,   219,   220,   220,   221,   221,
     222,   222,   222,   224,   223,   225,   225,   226,   226,   227,
     226,   228,   228,   229,   229,   230,   231,   231,   231,   232,
     232,   233,   233,   233,   233,   233,   233,   235,   234,   236,
     236,   237,   238,   238,   239,   239,   240,   240,   242,   241,
     243,   243,   244,   244,   244,   244,   245,   245,   247,   246,
     248,   248,   249,   249,   250,   250,   250,   250,   250,   252,
     251,   253,   253,   253,   253,   253,   254,   255,   255,   255,
     256,   256,   256,   256,   256,   256,   256,   257,   257,   257,
     257,   257,   257,   257,   258,   259,   259,   259,   260,   260,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   262,   262,   262,   262,   262,
     263,   263,   264,   263,   266,   265,   268,   267,   269,   269,
     269,   270,   270,   270,   272,   271,   273,   273,   273,   273,
     274,   274,   274,   275,   275,   275,   276,   276,   276,   277,
     277,   278,   278,   279,   279,   280,   280,   281,   281
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     1,     3,     3,     3,     2,
       2,     4,     6,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     5,     3,     3,     0,     5,     2,     3,     3,
       3,     3,     0,     7,     3,     3,     0,     1,     0,     6,
       2,     3,     3,     3,     3,     0,     1,     0,     3,     2,
       3,     0,     4,     3,     3,     0,     2,     2,     3,     2,
       3,     3,     4,     4,     5,     3,     3,     0,     6,     4,
       4,     3,     0,     2,     2,     0,     2,     1,     1,     1,
       1,     3,     3,     3,     2,     3,     2,     3,     3,     1,
       0,     6,     3,     3,     0,     6,     3,     3,     0,     6,
       3,     3,     0,     6,     3,     3,     0,     2,     3,     1,
       0,     5,     0,     5,     0,     5,     0,     5,     0,     0,
       3,     0,     1,     2,     2,     2,     1,     3,     1,     1,
       1,     3,     1,     0,     6,     3,     3,     0,     2,     3,
       1,     0,     5,     0,     5,     0,     5,     0,     5,     1,
       3,     0,     1,     1,     3,     0,     6,     2,     3,     0,
       1,     1,     2,     2,     2,     4,     3,     5,     3,     1,
       3,     1,     1,     3,     3,     5,     2,     5,     0,     7,
       3,     5,     0,     6,     2,     0,     1,     3,     1,     0,
       0,     5,     0,     3,     2,     3,     2,     3,     3,     3,
       5,     5,     3,     2,     3,     3,     1,     3,     3,     5,
       5,     7,     7,     7,     7,     4,     4,     4,     4,     6,
       6,     3,     4,     4,     1,     3,     3,     3,     1,     3,
       3,     3,     3,     0,     7,     3,     2,     0,     3,     0,
       5,     0,     2,     1,     3,     2,     0,     2,     3,     0,
       2,     1,     1,     1,     1,     1,     1,     0,     7,     5,
       4,     4,     4,     4,     0,     2,     1,     1,     0,     7,
       0,     2,     1,     4,     1,     2,     1,     1,     0,     7,
       2,     2,     0,     2,     1,     1,     1,     1,     1,     0,
       4,     1,     1,     2,     3,     3,     1,     2,     3,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       1,     1,     2,     2,     2,     3,     3,     3,     2,     3,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     2,     3,     2,     2,     2,     2,     3,     3,     3,
       3,     3,     3,     3,     3,     2,     3,     3,     3,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     1,     1,
       1,     1,     1,     1,     2,     1,     4,     5,     3,     1,
       1,     3,     5,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     4,     4,     5,     7,
       4,     3,     0,     6,     0,     6,     0,     6,     4,     3,
       2,     2,     2,     2,     0,     6,     5,     5,     4,     3,
       2,     3,     3,     2,     3,     3,     3,     3,     4,     1,
       3,     1,     3,     0,     1,     1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    25,   334,   335,   344,   336,
     330,   331,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   346,   347,     0,     0,     0,     0,     0,   319,
       0,     0,     0,     0,     0,     0,     0,   444,     0,     0,
       0,     0,   332,   333,   119,     0,     0,   436,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     4,     5,     8,    14,    23,    33,
      34,    55,     0,    35,    39,    68,     0,    40,    41,     0,
      36,    47,    48,    49,    37,   132,    38,   163,    42,   185,
      43,    10,   219,     0,     0,    46,    15,    16,    17,    24,
       9,    11,    13,    12,    45,    44,   350,   345,   351,   459,
     410,   401,   398,   399,   400,   402,   405,   403,   409,     0,
      29,     0,     0,     6,     0,   344,     0,    50,     0,   344,
     434,     0,     0,    87,     0,    89,     0,     0,     0,     0,
     465,     0,     0,     0,     0,     0,     0,     0,   187,     0,
       0,     0,     0,   263,     0,   308,     0,   327,     0,     0,
       0,     0,     0,     0,     0,   401,     0,     0,     0,   233,
     236,     0,     0,     0,     0,     0,     0,     0,     0,   258,
       0,   214,     0,     0,     0,     0,   453,   461,     0,     0,
      62,     0,   298,     0,     0,   450,     0,   459,     0,     0,
       0,   385,     0,   116,   459,     0,   392,   391,   358,     0,
     114,     0,   397,   396,   395,   394,   393,   371,   353,   352,
     354,     0,   376,   374,   390,   389,    85,     0,     0,     0,
      57,    85,    70,    97,     0,   136,   167,    85,     0,    85,
     220,   222,   206,     0,     0,    30,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   348,   348,   348,   348,   348,   348,
     348,   348,   348,   348,   348,   375,   373,   404,     0,     0,
       0,    18,   342,   343,   337,   338,   339,   340,     0,   341,
       0,   359,    54,    53,     0,     0,    59,    61,    58,    60,
      88,    91,    90,    72,    74,    71,    73,    95,     0,     0,
       0,   135,   134,     7,   166,   165,   188,   184,   204,   203,
      28,     0,    27,     0,   329,   328,   322,   326,     0,     0,
      22,    20,    21,   229,   228,   232,     0,   235,   234,     0,
     251,     0,     0,     0,     0,   238,     0,     0,   257,     0,
     256,     0,    26,     0,   215,   219,   219,   112,   111,   455,
     454,   464,     0,    65,    85,    64,     0,   424,   425,     0,
     456,     0,     0,   452,   451,     0,   457,     0,     0,   218,
       0,   216,   219,   118,   115,   117,   113,   356,   355,   357,
       0,     0,     0,    96,     0,     0,     0,     0,   224,   226,
       0,    85,     0,   210,   212,     0,   431,     0,     0,     0,
     408,   418,   422,   423,   421,   420,   419,   417,   416,   415,
     414,   413,   411,   449,     0,   384,   383,   382,   381,   380,
     379,   372,   377,   378,   388,   387,   386,   349,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     460,   253,    19,   252,     0,    51,    93,     0,   466,     0,
      92,     0,   215,   279,   271,     0,     0,     0,   312,   320,
       0,   323,     0,     0,   237,   245,   247,     0,   248,     0,
     246,     0,     0,   255,   260,   261,   262,   259,   440,     0,
      85,    85,   462,     0,   300,   427,   426,     0,   467,   458,
       0,   443,   442,   441,     0,    85,     0,    86,     0,     0,
       0,    77,    76,    81,     0,     0,     0,   107,     0,     0,
     108,   109,   110,     0,   139,     0,     0,   137,     0,   149,
       0,   170,     0,     0,   168,     0,     0,   190,   191,    85,
     225,   227,     0,     0,   223,     0,   208,     0,   432,   430,
       0,   406,     0,   448,     0,   368,   367,   366,   361,   360,
     364,   363,   362,   365,   370,   369,    31,     0,     0,     0,
      94,   266,     0,     0,     0,   311,   276,   272,   273,   310,
       0,   325,   324,   231,   230,     0,     0,   239,     0,     0,
     240,     0,     0,   439,     0,     0,     0,    66,     0,     0,
     428,     0,   217,     0,    56,     0,    79,     0,     0,     0,
      85,    85,     0,   120,     0,     0,   124,     0,     0,   128,
       0,     0,   106,   138,     0,   162,   160,   158,   159,     0,
     156,   153,     0,     0,   169,     0,   182,   183,     0,   179,
       0,     0,   194,   201,   202,     0,     0,   199,     0,   192,
       0,   205,     0,     0,     0,   207,   211,     0,   407,   412,
     447,   446,     0,    52,   101,     0,   269,   268,   281,     0,
       0,     0,     0,     0,     0,   282,   286,   280,   285,   283,
     284,   294,   265,     0,   275,     0,   314,     0,   315,   318,
     317,   316,   313,   249,   250,     0,     0,     0,     0,   438,
     435,   445,     0,    67,   302,     0,     0,   304,   301,     0,
     468,   437,    80,    84,    83,    69,     0,     0,   123,    85,
     122,   127,    85,   126,   131,    85,   130,    98,   144,   146,
       0,   140,   142,     0,   133,    85,     0,   150,   175,   177,
     171,   173,   181,   164,   198,     0,   196,     0,     0,   186,
     221,   213,     0,   433,    32,   100,     0,    99,     0,     0,
     264,   287,     0,     0,     0,     0,   277,     0,   274,   309,
     241,   243,   242,   244,    63,   306,     0,   307,   305,   299,
     429,    82,     0,     0,     0,    85,     0,   161,    85,     0,
     157,     0,   155,    85,     0,    85,     0,   180,   195,   200,
       0,   209,   104,   103,   270,     0,   219,     0,     0,     0,
     296,     0,   297,   295,   278,     0,     0,     0,     0,     0,
     147,     0,   143,     0,   178,     0,   174,   197,   290,    85,
       0,   292,   293,   291,   303,   121,   125,   129,     0,   289,
       0,   288
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    64,    65,   299,    66,   517,    68,   126,
      69,    70,   226,    71,    72,    73,   374,   712,    74,   231,
      75,    76,   520,   620,   521,   522,   621,   523,   400,    77,
      78,    79,    80,   402,   466,   767,   528,   529,    81,    82,
      83,   530,   729,   531,   732,   532,   735,    84,   235,    85,
     404,   537,   798,   799,   795,   796,   538,   643,   539,   747,
     639,   640,    86,   236,    87,   405,   544,   805,   806,   803,
     804,   648,   649,    88,   237,    89,   546,   547,   548,   549,
     656,   657,    90,    91,    92,   664,    93,   555,    94,   390,
     391,   239,   411,   412,   240,   241,    95,    96,    97,   171,
      98,    99,   175,   100,   178,   179,   101,   331,   473,   474,
     768,   477,   587,   588,   694,   583,   687,   688,   816,   689,
     690,   691,   775,   823,   102,   376,   608,   718,   788,   103,
     333,   478,   590,   702,   104,   159,   338,   339,   105,   106,
     300,   107,   108,   448,   109,   110,   111,   667,   112,   182,
     113,   200,   365,   392,   114,   183,   115,   116,   117,   118,
     119,   188,   372,   141,   199
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -562
static const yytype_int16 yypact[] =
{
    -562,    71,   854,  -562,    72,  -562,  -562,  -562,   143,  -562,
    -562,  -562,   167,    13,  3597,   419,   327,  3669,    82,  3741,
     117,  3813,  -562,  -562,   287,  3885,   449,   515,  3381,  -562,
     529,  3957,   522,   541,   395,   544,   294,  -562,  4029,  5467,
     350,   127,  -562,  -562,  -562,  5827,  5300,  -562,  5827,  3453,
    5827,  5827,  5827,  3525,  5827,  5827,  5827,  5827,  5827,  5827,
     311,  5827,  5827,   458,  -562,  -562,  -562,  -562,  -562,  -562,
    -562,  -562,  3309,  -562,  -562,  -562,  3309,  -562,  -562,     6,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
    -562,  -562,   158,  3309,   263,  -562,  -562,  -562,  -562,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  4834,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,   -51,
    -562,   115,  5827,  -562,   183,  -562,     2,   192,    28,   228,
    -562,  4657,   322,  -562,   330,  -562,   337,   313,  4730,   349,
     324,    84,   404,  4886,   420,   451,  4938,   455,  -562,  3309,
     462,  4990,   483,  -562,   494,  -562,   517,  -562,  5042,   547,
     543,   576,   577,   581,  6378,   583,   586,   338,   587,  -562,
    -562,   193,   588,   152,   425,   191,   589,   401,   194,  -562,
     590,  -562,   293,   293,   592,  5094,  -562,  6378,   554,   594,
    -562,  3309,  -562,  6064,  5539,  -562,   556,  5888,    64,   151,
     139,  6628,   605,  -562,  6378,   195,   473,   473,   240,   612,
    -562,   200,   240,   240,   240,   240,   240,   240,  -562,  -562,
    -562,   513,   240,   240,  -562,  -562,  -562,   615,   616,   300,
    -562,  -562,  -562,  -562,  3309,  -562,  -562,  -562,   413,  -562,
    -562,  -562,  -562,   617,   140,  -562,  5611,  5395,   613,  5827,
    5827,  5827,  5827,  5827,  5827,  5827,  5827,  5827,  5827,  5827,
    5827,  4101,  5827,  5827,  5827,  5827,  5827,  5827,  5827,  5827,
    5827,   620,  5827,  5827,   618,   618,   618,   618,   618,   618,
     618,   618,   618,   618,   618,  -562,  -562,  -562,  5827,  5827,
     624,  -562,  -562,  -562,  -562,  -562,  -562,  -562,   614,  -562,
     626,  6378,  -562,  -562,   628,  5827,  -562,  -562,  -562,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  5827,   628,
    4173,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
    -562,   378,  -562,   489,  -562,  -562,  -562,  -562,   204,   148,
    -562,  -562,  -562,  -562,  -562,  -562,   130,  -562,  -562,   629,
    -562,   627,    31,   209,   634,  -562,   531,   632,  -562,   165,
    -562,   633,  -562,   638,   636,   158,   158,  -562,  -562,  -562,
    -562,  -562,  5827,  -562,  -562,  -562,   643,  -562,  -562,  6116,
    -562,  5683,  5827,  -562,  -562,   593,  -562,  5827,   591,  -562,
     199,  -562,   158,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
    1801,    50,   989,  -562,   599,   630,  1569,   336,  -562,  -562,
    1917,  -562,  3309,  -562,  -562,   219,  -562,   224,  5827,  5950,
    -562,  6378,  6378,  6378,  6378,  6378,  6378,  6378,  6378,  6378,
    6378,  6378,  6428,  -562,  4584,  6578,  6628,  5273,  5273,  5273,
    5273,  5273,  5273,  5273,  -562,   473,   473,  -562,  5827,  5827,
    5827,  5827,  5827,  5827,  5827,  5827,  5827,  5827,  5827,  4782,
    6478,  -562,  -562,  -562,   574,  6378,  -562,  6168,  -562,   648,
    6378,   653,   636,  -562,   625,   656,   660,   668,  -562,  -562,
     551,  -562,   669,   670,  -562,  -562,  -562,   671,  -562,   672,
    -562,    20,    25,  -562,  -562,  -562,  -562,  -562,  -562,   226,
    -562,  -562,  6378,  2033,  -562,  -562,  -562,  6012,  6378,  -562,
    6222,  -562,  -562,  -562,   636,  -562,   673,  -562,   500,  4245,
     665,  -562,  -562,  -562,   474,   490,   495,  -562,   674,   989,
    -562,  -562,  -562,   676,  -562,    86,   499,  -562,   689,  -562,
     678,  -562,   187,   700,  -562,   174,   704,   652,  -562,  -562,
    -562,  -562,   680,  2149,  -562,   631,  -562,   375,  -562,  -562,
    6274,  -562,  5827,  -562,  4317,   410,   410,   262,   288,   288,
     268,   268,   268,   390,   240,   240,  -562,  5827,  5827,  4389,
    -562,  -562,   234,   277,   712,  -562,   661,   640,  -562,  -562,
     415,  -562,  -562,  -562,  -562,   717,   718,  -562,   716,   721,
    -562,   722,   724,  -562,   720,  2265,  2381,  5827,   326,  5827,
    -562,  5827,  -562,  2497,  -562,   729,  -562,   731,  5146,   732,
    -562,  -562,   734,  -562,  3309,   735,  -562,  3309,   736,  -562,
    3309,   737,  -562,  -562,   382,  -562,  -562,  -562,   647,   265,
    -562,  -562,   738,   383,  -562,   403,  -562,  -562,   267,  -562,
     739,   740,  -562,  -562,  -562,   628,   124,  -562,   742,  -562,
    1685,  -562,   743,   705,   693,  -562,  -562,   695,  -562,   677,
    -562,  6528,   206,  6378,  -562,  4522,  -562,  -562,  -562,   354,
     748,   749,   761,   763,   304,  -562,  -562,  -562,  -562,  -562,
    -562,  -562,  -562,  5755,  -562,   660,  -562,   751,  -562,  -562,
    -562,  -562,  -562,  -562,  -562,   767,   768,   769,   772,  -562,
    -562,  -562,   774,  6378,  -562,   259,   775,  -562,  -562,  6326,
    6378,  -562,  -562,  -562,  -562,  -562,  2613,    50,  -562,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
       7,  -562,  -562,   107,  -562,  -562,  3309,  -562,  -562,  -562,
    -562,  -562,   550,  -562,  -562,   777,  -562,   579,   628,  -562,
    -562,  -562,   778,  -562,  -562,  -562,  4461,  -562,   728,  5827,
    -562,  -562,   709,   733,   741,   303,  -562,   144,  -562,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,   121,  -562,  -562,  -562,
    -562,  -562,  2729,  2845,  2961,  -562,  3309,  -562,  -562,  3309,
    -562,  3077,  -562,  -562,  3309,  -562,  3309,  -562,  -562,  -562,
     782,  -562,  -562,  6378,  -562,  5198,   158,   121,   783,   785,
    -562,   788,  -562,  -562,  -562,   208,   789,   790,   792,  1105,
    -562,  1221,  -562,  1337,  -562,  1453,  -562,  -562,  -562,  -562,
     793,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  3193,  -562,
     796,  -562
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -562,  -562,  -562,  -562,  -562,  -356,  -562,    -2,  -562,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
    -562,  -562,    73,  -562,  -562,  -562,  -562,  -562,   -16,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,   272,  -562,  -562,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,   398,  -562,  -562,  -562,
    -562,    62,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
    -562,  -562,    54,  -562,  -562,  -562,  -562,  -562,   260,  -562,
    -562,    51,  -562,  -561,  -562,  -562,  -562,  -562,  -562,  -214,
     295,  -345,  -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
    -562,  -175,  -562,  -562,  -562,   450,  -562,  -562,  -562,  -562,
    -562,   339,  -562,   119,  -562,  -562,  -562,   220,  -562,   222,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,  -562,   -10,  -562,
    -562,  -562,  -562,  -562,  -562,  -562,  -562,   340,  -562,  -562,
    -330,   -11,  -562,   482,   -13,   266,   786,  -562,  -562,  -562,
    -562,  -562,   635,  -562,  -562,  -562,  -562,  -562,  -562,  -562,
     -36,  -562,  -562,  -562,  -562
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -464
static const yytype_int16 yytable[] =
{
      67,   131,   127,   496,   138,   303,   143,   140,   146,   233,
     198,   291,   151,   205,   124,   158,   483,   211,   164,   125,
     500,   501,   685,   597,   288,   185,   187,   289,   600,   698,
     415,   306,   193,   197,   486,   201,   204,   206,   207,   208,
     204,   212,   213,   214,   215,   216,   217,   515,   222,   223,
     234,     4,   225,     5,     6,     7,     8,     9,    10,    11,
     -75,    13,    14,    15,    16,   383,    17,   518,   519,    18,
     230,     3,   307,    19,   232,   120,    21,    22,    23,    24,
     304,    25,   227,   139,   228,    28,    29,   634,   125,    31,
     291,   242,   635,   636,   637,   229,   598,    37,    38,    39,
      40,   601,    42,    43,    44,   298,    45,   487,    46,   301,
      47,   291,   599,   635,   636,   637,   290,   602,   144,   291,
     292,   384,   293,   294,   295,   291,   292,   756,   293,   294,
     295,   482,    48,   192,   291,   292,    49,   293,   294,   295,
     388,   414,   289,    50,    51,   389,   389,   327,    52,   481,
     499,  -321,   385,   351,    53,  -254,    54,    55,    56,    57,
      58,    59,   319,    60,    61,    62,    63,   296,   297,   291,
     123,   494,   495,   296,   297,   651,   320,   652,   653,   638,
     654,   379,   296,   297,   298,  -254,   302,   121,   645,   375,
    -181,   646,   354,   647,   355,  -215,   348,   360,   394,   824,
     512,  -215,   757,   396,   238,   298,   122,   479,   386,   764,
     417,   785,   488,   298,   822,   401,   758,  -215,  -215,   298,
     556,   406,   289,   410,   356,   558,  -321,   603,   298,   387,
    -254,  -181,   403,   204,   419,   676,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   432,   434,   435,
     436,   437,   438,   439,   440,   441,   442,   443,   582,   445,
     446,   513,   785,   298,   243,  -181,   655,   305,   741,   357,
     750,   349,   361,   289,   557,   459,   460,   514,   289,   559,
     678,   604,   480,   679,   289,   489,   787,   680,   147,   677,
     148,   122,   465,   464,   363,   180,   246,   514,   247,   248,
     181,   180,   289,   121,   514,   467,   820,   470,   468,   742,
     773,   751,   514,   821,   681,   218,   313,   219,   246,   244,
     247,   248,   682,   683,   246,   310,   247,   248,   134,   714,
     135,   149,   715,   311,   786,   684,   716,   787,   220,   550,
     312,   774,   136,   743,   246,   752,   247,   248,   682,   364,
    -434,   189,   317,   190,   285,   286,  -434,   314,   503,   502,
     277,   278,   279,   280,   281,   282,   283,   284,   507,   508,
     221,   282,   283,   284,   510,   287,   285,   286,   665,   471,
     551,  -267,   285,   286,   797,   738,   745,   638,   279,   280,
     281,   282,   283,   284,   191,   553,   172,   287,   121,   318,
     527,   173,   285,   286,   287,   560,   748,   321,   686,   287,
     554,  -267,   287,   346,   407,   699,   408,   287,   696,   666,
     132,   679,   133,   323,   287,   697,   739,   746,   174,   769,
     287,   352,   353,   717,   472,   565,   566,   567,   568,   569,
     570,   571,   572,   573,   574,   575,   246,   749,   247,   248,
     152,   287,   681,   287,   324,   153,   825,   409,   326,   287,
     682,   683,   224,   287,   125,   328,   246,   287,   247,   248,
     287,   839,   287,   287,   287,   622,   359,   623,   287,   287,
     287,   287,   287,   287,   605,   606,   330,   840,   287,   287,
     475,   625,  -271,   626,   283,   284,   628,   332,   629,   613,
     641,   615,  -152,   616,   285,   286,   618,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   154,   397,   624,   398,
     334,   155,   476,   166,   285,   286,   669,   527,   167,   246,
     160,   247,   248,   660,   627,   161,   162,   491,   492,   630,
     399,   672,   168,  -152,   169,   176,   340,   170,   336,   204,
     177,   671,   591,   337,   646,   369,   647,   337,  -463,  -463,
    -463,  -463,  -463,  -463,   204,   673,   675,   287,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   341,
     342,  -463,  -463,   653,   343,   654,   344,   285,   286,   345,
     347,   350,   358,   362,   713,   367,   719,   373,   720,  -463,
     533,  -463,   534,  -463,   726,   727,  -463,  -463,   393,  -148,
    -463,   370,  -463,   380,  -463,   395,   152,   154,   462,   420,
     413,   447,   730,   535,   536,   733,   444,   461,   736,   463,
     485,   540,   371,   541,   125,   484,  -463,   490,   493,   177,
    -148,   498,   389,  -151,   755,   287,   504,  -463,  -463,   578,
     509,   580,  -463,   511,   542,   536,   581,   777,   476,   585,
    -463,  -463,  -463,  -463,  -463,  -463,   586,  -463,  -463,  -463,
    -463,   589,   593,   594,  -151,   619,   614,   595,   596,   633,
     204,   644,   545,   661,   631,   287,   663,   287,   287,   287,
     287,   287,   287,   287,   287,   287,   287,   287,   287,   642,
     287,   287,   287,   287,   287,   287,   287,   287,   287,   287,
     650,   287,   287,   792,   658,   692,   793,   693,   695,   794,
     703,   704,   705,   709,   740,   287,   287,   706,   707,   801,
     708,   287,   722,   287,   723,   725,   287,   728,   731,   734,
     737,   744,   753,   754,   802,   759,   760,   810,   762,   761,
     763,   770,   771,   813,   779,   289,   815,   449,   450,   451,
     452,   453,   454,   455,   456,   457,   458,   181,   287,   772,
     780,   781,   782,   287,   287,   783,   287,   784,   789,   829,
     808,   811,   831,   814,   817,   837,   841,   833,   842,   835,
     818,   843,   845,   846,   830,   847,   849,   832,   819,   851,
     791,   632,   834,   543,   836,   800,   807,   659,   809,   612,
     700,   497,   701,   584,   778,   844,     0,   165,   366,     0,
     592,     0,     0,   848,     0,     0,   287,     0,     0,     0,
       0,   287,   287,   287,   287,   287,   287,   287,   287,   287,
     287,   287,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    -2,     4,     0,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,     0,
      17,     0,     0,    18,     0,     0,     0,    19,    20,     0,
      21,    22,    23,    24,   287,    25,    26,     0,    27,    28,
      29,     0,    30,    31,    32,    33,    34,    35,     0,    36,
       0,    37,    38,    39,    40,    41,    42,    43,    44,     0,
      45,     0,    46,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,   287,     0,   287,
      49,   287,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,    52,     0,     0,     0,     0,     0,    53,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,     0,     0,     0,     0,     0,     0,     0,     0,   287,
       0,     0,     0,     0,     0,   287,   287,     0,     0,     0,
       4,     0,     5,     6,     7,     8,     9,    10,    11,  -105,
      13,    14,    15,    16,     0,    17,     0,     0,    18,   524,
     525,   526,    19,     0,     0,    21,    22,    23,    24,     0,
      25,   227,     0,   228,    28,    29,     0,     0,    31,     0,
       0,     0,     0,     0,   229,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,   287,
       0,   287,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,     4,     0,     5,     6,
       7,     8,     9,    10,    11,  -145,    13,    14,    15,    16,
       0,    17,     0,     0,    18,     0,     0,     0,    19,  -145,
    -145,    21,    22,    23,    24,     0,    25,   227,     0,   228,
      28,    29,     0,     0,    31,     0,     0,     0,     0,  -145,
     229,     0,    37,    38,    39,    40,     0,    42,    43,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,     4,     0,     5,     6,     7,     8,     9,    10,
      11,  -141,    13,    14,    15,    16,     0,    17,     0,     0,
      18,     0,     0,     0,    19,  -141,  -141,    21,    22,    23,
      24,     0,    25,   227,     0,   228,    28,    29,     0,     0,
      31,     0,     0,     0,     0,  -141,   229,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,  -176,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,  -176,  -176,    21,    22,    23,    24,     0,    25,   227,
       0,   228,    28,    29,     0,     0,    31,     0,     0,     0,
       0,  -176,   229,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,     4,     0,     5,     6,     7,     8,
       9,    10,    11,  -172,    13,    14,    15,    16,     0,    17,
       0,     0,    18,     0,     0,     0,    19,  -172,  -172,    21,
      22,    23,    24,     0,    25,   227,     0,   228,    28,    29,
       0,     0,    31,     0,     0,     0,     0,  -172,   229,     0,
      37,    38,    39,    40,     0,    42,    43,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,     0,     0,     0,     0,     0,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
       4,     0,     5,     6,     7,     8,     9,    10,    11,  -189,
      13,    14,    15,    16,     0,    17,     0,     0,    18,     0,
       0,     0,    19,     0,     0,    21,    22,    23,    24,   545,
      25,   227,     0,   228,    28,    29,     0,     0,    31,     0,
       0,     0,     0,     0,   229,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,     4,     0,     5,     6,
       7,     8,     9,    10,    11,  -193,    13,    14,    15,    16,
       0,    17,     0,     0,    18,     0,     0,     0,    19,     0,
       0,    21,    22,    23,    24,  -193,    25,   227,     0,   228,
      28,    29,     0,     0,    31,     0,     0,     0,     0,     0,
     229,     0,    37,    38,    39,    40,     0,    42,    43,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,     4,     0,     5,     6,     7,     8,     9,    10,
      11,   516,    13,    14,    15,    16,     0,    17,     0,     0,
      18,     0,     0,     0,    19,     0,     0,    21,    22,    23,
      24,     0,    25,   227,     0,   228,    28,    29,     0,     0,
      31,     0,     0,     0,     0,     0,   229,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,   552,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,     0,     0,    21,    22,    23,    24,     0,    25,   227,
       0,   228,    28,    29,     0,     0,    31,     0,     0,     0,
       0,     0,   229,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,     4,     0,     5,     6,     7,     8,
       9,    10,    11,   607,    13,    14,    15,    16,     0,    17,
       0,     0,    18,     0,     0,     0,    19,     0,     0,    21,
      22,    23,    24,     0,    25,   227,     0,   228,    28,    29,
       0,     0,    31,     0,     0,     0,     0,     0,   229,     0,
      37,    38,    39,    40,     0,    42,    43,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,     0,     0,     0,     0,     0,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
       4,     0,     5,     6,     7,     8,     9,    10,    11,   662,
      13,    14,    15,    16,     0,    17,     0,     0,    18,     0,
       0,     0,    19,     0,     0,    21,    22,    23,    24,     0,
      25,   227,     0,   228,    28,    29,     0,     0,    31,     0,
       0,     0,     0,     0,   229,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,     4,     0,     5,     6,
       7,     8,     9,    10,    11,   710,    13,    14,    15,    16,
       0,    17,     0,     0,    18,     0,     0,     0,    19,     0,
       0,    21,    22,    23,    24,     0,    25,   227,     0,   228,
      28,    29,     0,     0,    31,     0,     0,     0,     0,     0,
     229,     0,    37,    38,    39,    40,     0,    42,    43,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,     4,     0,     5,     6,     7,     8,     9,    10,
      11,   711,    13,    14,    15,    16,     0,    17,     0,     0,
      18,     0,     0,     0,    19,     0,     0,    21,    22,    23,
      24,     0,    25,   227,     0,   228,    28,    29,     0,     0,
      31,     0,     0,     0,     0,     0,   229,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,     0,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,     0,     0,    21,    22,    23,    24,     0,    25,   227,
       0,   228,    28,    29,     0,     0,    31,     0,     0,     0,
       0,     0,   229,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,   721,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,     4,     0,     5,     6,     7,     8,
       9,    10,    11,   -78,    13,    14,    15,    16,     0,    17,
       0,     0,    18,     0,     0,     0,    19,     0,     0,    21,
      22,    23,    24,     0,    25,   227,     0,   228,    28,    29,
       0,     0,    31,     0,     0,     0,     0,     0,   229,     0,
      37,    38,    39,    40,     0,    42,    43,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,     0,     0,     0,     0,     0,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
       4,     0,     5,     6,     7,     8,     9,    10,    11,   826,
      13,    14,    15,    16,     0,    17,     0,     0,    18,     0,
       0,     0,    19,     0,     0,    21,    22,    23,    24,     0,
      25,   227,     0,   228,    28,    29,     0,     0,    31,     0,
       0,     0,     0,     0,   229,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,     0,     0,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,     4,     0,     5,     6,
       7,     8,     9,    10,    11,   827,    13,    14,    15,    16,
       0,    17,     0,     0,    18,     0,     0,     0,    19,     0,
       0,    21,    22,    23,    24,     0,    25,   227,     0,   228,
      28,    29,     0,     0,    31,     0,     0,     0,     0,     0,
     229,     0,    37,    38,    39,    40,     0,    42,    43,    44,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
       0,    49,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,     0,     0,     0,     0,     0,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,     4,     0,     5,     6,     7,     8,     9,    10,
      11,   828,    13,    14,    15,    16,     0,    17,     0,     0,
      18,     0,     0,     0,    19,     0,     0,    21,    22,    23,
      24,     0,    25,   227,     0,   228,    28,    29,     0,     0,
      31,     0,     0,     0,     0,     0,   229,     0,    37,    38,
      39,    40,     0,    42,    43,    44,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,    49,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,    53,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,     4,     0,
       5,     6,     7,     8,     9,    10,    11,  -154,    13,    14,
      15,    16,     0,    17,     0,     0,    18,     0,     0,     0,
      19,     0,     0,    21,    22,    23,    24,     0,    25,   227,
       0,   228,    28,    29,     0,     0,    31,     0,     0,     0,
       0,     0,   229,     0,    37,    38,    39,    40,     0,    42,
      43,    44,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,    49,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,     4,     0,     5,     6,     7,     8,
       9,    10,    11,   850,    13,    14,    15,    16,     0,    17,
       0,     0,    18,     0,     0,     0,    19,     0,     0,    21,
      22,    23,    24,     0,    25,   227,     0,   228,    28,    29,
       0,     0,    31,     0,     0,     0,     0,     0,   229,     0,
      37,    38,    39,    40,     0,    42,    43,    44,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,     0,    49,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,     0,     0,     0,     0,     0,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
       4,     0,     5,     6,     7,     8,     9,    10,    11,     0,
      13,    14,    15,    16,     0,    17,     0,     0,    18,     0,
       0,     0,    19,     0,     0,    21,    22,    23,    24,     0,
      25,   227,     0,   228,    28,    29,     0,     0,    31,     0,
       0,     0,     0,     0,   229,     0,    37,    38,    39,    40,
       0,    42,    43,    44,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   156,     0,   157,     6,     7,   129,     9,    10,
      11,    48,     0,     0,     0,    49,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,    53,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   130,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   202,     0,   203,     6,     7,   129,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   130,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   209,     0,   210,     6,
       7,   129,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,     0,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     130,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   128,     0,
       0,     6,     7,   129,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   130,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     137,     0,     0,     6,     7,   129,     9,    10,    11,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   130,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   142,     0,     0,     6,     7,   129,     9,    10,
      11,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   130,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   145,     0,     0,     6,     7,   129,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   130,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   150,     0,     0,     6,
       7,   129,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,     0,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     130,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   163,     0,
       0,     6,     7,   129,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   130,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     184,     0,     0,     6,     7,   129,     9,    10,    11,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   130,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   433,     0,     0,     6,     7,   129,     9,    10,
      11,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   130,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   469,     0,     0,     6,     7,   129,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   130,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,    45,
       0,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   617,     0,     0,     6,
       7,   129,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,     0,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     130,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   670,     0,
       0,     6,     7,   129,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   130,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     674,     0,     0,     6,     7,   129,     9,    10,    11,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   130,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,    45,     0,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   812,     0,     0,     6,     7,   129,     9,    10,
      11,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   130,     0,    37,     0,
      39,     0,     0,    42,    43,     0,     0,    45,     0,    46,
       0,    47,     0,   765,     0,  -102,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
       0,     0,     0,     0,     0,     0,  -102,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   246,     0,
     247,   248,     0,     0,     0,   563,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,     0,     0,
     766,   261,   262,   263,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,     0,     0,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   564,     0,
       0,     0,     0,     0,     0,     0,   285,   286,     0,     0,
     246,     0,   247,   248,     0,     0,     0,     0,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     308,     0,     0,   261,   262,   263,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,     0,     0,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
       0,     0,     0,     0,     0,     0,     0,     0,   285,   286,
       0,   309,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   246,     0,   247,   248,     0,     0,     0,
       0,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   315,     0,     0,   261,   262,   263,     0,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
       0,     0,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,     0,     0,     0,     0,     0,     0,     0,
       0,   285,   286,     0,   316,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   576,   246,     0,   247,   248,
       0,     0,     0,     0,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,     0,     0,     0,   261,
     262,   263,     0,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,     0,     0,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,     0,   245,   246,     0,
     247,   248,     0,     0,   285,   286,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,     0,     0,
     577,   261,   262,   263,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,     0,     0,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,     0,   322,
     246,     0,   247,   248,     0,     0,   285,   286,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
       0,     0,     0,   261,   262,   263,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,     0,     0,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
       0,   325,   246,     0,   247,   248,     0,     0,   285,   286,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,     0,
       0,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,     0,   329,   246,     0,   247,   248,     0,     0,
     285,   286,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,     0,   335,   246,     0,   247,   248,
       0,     0,   285,   286,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,     0,     0,     0,   261,
     262,   263,     0,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,     0,     0,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,     0,   368,   246,     0,
     247,   248,     0,     0,   285,   286,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,     0,     0,
       0,   261,   262,   263,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,     0,     0,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,     0,   724,
     246,     0,   247,   248,     0,     0,   285,   286,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
       0,     0,     0,   261,   262,   263,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,     0,     0,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
       0,   838,   246,     0,   247,   248,     0,     0,   285,   286,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,     0,
       0,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,     0,     0,   246,     0,   247,   248,     0,     0,
     285,   286,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,     6,     7,   129,     9,    10,    11,
       0,     0,   285,   286,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    22,    23,   246,
       0,   247,   248,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   194,   130,     0,    37,     0,    39,
       0,     0,    42,    43,     0,     0,    45,   195,    46,     0,
      47,     0,   196,   271,   272,   273,     0,     0,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,     0,
       0,     0,    48,     0,     0,     0,     0,   285,   286,     0,
       0,     0,     0,    50,    51,     0,     0,     0,    52,     6,
       7,   129,     9,    10,    11,     0,    54,    55,    56,    57,
      58,    59,     0,    60,    61,    62,    63,     0,     0,     0,
       0,     0,    22,    23,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   194,
     130,     0,    37,     0,    39,     0,     0,    42,    43,     0,
       0,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     7,   129,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,   418,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   130,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,   186,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     6,     7,   129,     9,    10,    11,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,    22,    23,     0,     0,
       0,     0,     0,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   130,     0,    37,     0,    39,     0,
       0,    42,    43,     0,     0,    45,   378,    46,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     6,     7,   129,     9,    10,
      11,    48,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,    51,     0,     0,     0,    52,    22,    23,
       0,     0,     0,     0,     0,    54,    55,    56,    57,    58,
      59,     0,    60,    61,    62,    63,   130,     0,    37,     0,
      39,     0,     0,    42,    43,     0,   416,    45,     0,    46,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,   129,
       9,    10,    11,    48,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,    51,     0,     0,     0,    52,
      22,    23,     0,     0,     0,     0,     0,    54,    55,    56,
      57,    58,    59,     0,    60,    61,    62,    63,   130,     0,
      37,     0,    39,     0,     0,    42,    43,     0,     0,    45,
     506,    46,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,   129,     9,    10,    11,    48,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,    51,     0,     0,
       0,    52,    22,    23,     0,     0,     0,     0,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
     130,     0,    37,     0,    39,     0,     0,    42,    43,     0,
     776,    45,     0,    46,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     7,   129,     9,    10,    11,    48,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,    51,
       0,     0,     0,    52,    22,    23,     0,     0,     0,     0,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,   130,     0,    37,     0,    39,     0,     0,    42,
      43,     0,     0,    45,     0,    46,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,    51,     0,     0,     0,    52,     0,     0,     0,     0,
       0,     0,   381,    54,    55,    56,    57,    58,    59,     0,
      60,    61,    62,    63,   246,     0,   247,   248,     0,     0,
     382,     0,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   381,     0,     0,     0,     0,     0,
       0,     0,   285,   286,     0,     0,   246,   561,   247,   248,
       0,     0,     0,     0,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,     0,     0,     0,   261,
     262,   263,     0,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,     0,     0,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   609,     0,     0,     0,
       0,     0,     0,     0,   285,   286,     0,     0,   246,   610,
     247,   248,     0,     0,     0,     0,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,     0,     0,
       0,   261,   262,   263,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,     0,     0,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,     0,   377,
     246,     0,   247,   248,     0,     0,   285,   286,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
       0,     0,     0,   261,   262,   263,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,     0,     0,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
       0,     0,   246,   505,   247,   248,     0,     0,   285,   286,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,     0,
       0,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,     0,     0,   246,     0,   247,   248,     0,     0,
     285,   286,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,   579,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,     0,     0,     0,     0,   246,     0,
     247,   248,   285,   286,   611,     0,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,     0,     0,
       0,   261,   262,   263,     0,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,     0,     0,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,     0,     0,
     246,   668,   247,   248,     0,     0,   285,   286,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
       0,     0,     0,   261,   262,   263,     0,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,     0,     0,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
       0,     0,   246,   790,   247,   248,     0,     0,   285,   286,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,     0,     0,     0,   261,   262,   263,     0,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,     0,
       0,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,     0,     0,   246,     0,   247,   248,     0,     0,
     285,   286,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   246,     0,   247,   248,     0,     0,
       0,     0,   285,   286,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   260,     0,     0,   562,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   246,     0,   247,   248,     0,     0,
       0,     0,   285,   286,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   261,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   246,     0,   247,   248,     0,     0,
       0,     0,   285,   286,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   262,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   246,     0,   247,   248,     0,     0,
       0,     0,   285,   286,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   263,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   246,     0,   247,   248,     0,     0,
       0,     0,   285,   286,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,     0,     0,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,     0,     0,     0,     0,     0,     0,
       0,     0,   285,   286
};

static const yytype_int16 yycheck[] =
{
       2,    14,    13,   359,    17,     3,    19,    18,    21,     3,
      46,     4,    25,    49,     1,    28,   346,    53,    31,     6,
     365,   366,   583,     3,    75,    38,    39,    78,     3,   590,
     244,     3,    45,    46,     3,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,   392,    61,    62,
      44,     1,    63,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,     1,    16,    17,    18,    19,
      72,     0,    44,    23,    76,     3,    26,    27,    28,    29,
      78,    31,    32,     1,    34,    35,    36,     1,     6,    39,
       4,    93,     6,     7,     8,    45,    76,    47,    48,    49,
      50,    76,    52,    53,    54,    98,    56,    76,    58,   122,
      60,     4,    92,     6,     7,     8,     1,    92,     1,     4,
       5,    57,     7,     8,     9,     4,     5,     3,     7,     8,
       9,     1,    82,     6,     4,     5,    86,     7,     8,     9,
       1,     1,    78,    93,    94,     6,     6,   149,    98,     1,
     364,     3,     1,     1,   104,     3,   106,   107,   108,   109,
     110,   111,    78,   113,   114,   115,   116,    52,    53,     4,
       3,     6,     7,    52,    53,     1,    92,     3,     4,   535,
       6,   194,    52,    53,    98,    33,     3,    44,     1,   191,
       3,     4,     1,     6,     3,    55,     3,     3,     3,    55,
       1,    62,    78,     3,    46,    98,    63,     3,    57,     3,
     246,     3,     3,    98,   775,   231,    92,    78,    78,    98,
       1,   237,    78,   239,    33,     1,    78,     1,    98,    78,
      78,    44,   234,   246,   247,     1,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   472,   272,
     273,    62,     3,    98,     1,    78,    92,    75,     3,    78,
       3,    78,    78,    78,    55,   288,   289,    78,    78,    55,
       3,    55,    78,     6,    78,    76,    78,    10,     1,    55,
       3,    63,   305,   304,     1,     1,    56,    78,    58,    59,
       6,     1,    78,    44,    78,   318,     3,   320,   319,    44,
       6,    44,    78,    10,    37,     4,     3,     6,    56,    56,
      58,    59,    45,    46,    56,     3,    58,    59,     1,     3,
       3,    44,     6,     3,    75,    58,    10,    78,    27,     3,
       3,    37,    15,    78,    56,    78,    58,    59,    45,    56,
      56,     1,     3,     3,   114,   115,    56,    44,   374,   372,
      98,    99,   100,   101,   102,   103,   104,   105,   381,   382,
      59,   103,   104,   105,   387,   109,   114,   115,     3,     1,
      44,     3,   114,   115,   740,     3,     3,   743,   100,   101,
     102,   103,   104,   105,    44,   411,     1,   131,    44,    75,
     402,     6,   114,   115,   138,   418,     3,     3,   583,   143,
     412,    33,   146,    75,     1,   590,     3,   151,     3,    44,
       1,     6,     3,     3,   158,    10,    44,    44,    33,    75,
     164,     6,     7,   608,    56,   448,   449,   450,   451,   452,
     453,   454,   455,   456,   457,   458,    56,    44,    58,    59,
       1,   185,    37,   187,     3,     6,   786,    44,     3,   193,
      45,    46,     4,   197,     6,     3,    56,   201,    58,    59,
     204,   816,   206,   207,   208,     1,    75,     3,   212,   213,
     214,   215,   216,   217,   500,   501,     3,   817,   222,   223,
       1,     1,     3,     3,   104,   105,     1,     3,     3,   515,
       1,     1,     3,     3,   114,   115,   519,    97,    98,    99,
     100,   101,   102,   103,   104,   105,     1,     4,    44,     6,
       3,     6,    33,     1,   114,   115,   562,   529,     6,    56,
       1,    58,    59,   549,    44,     6,     7,     6,     7,    44,
      27,   577,     1,    44,     3,     1,     3,     6,     1,   562,
       6,   564,     1,     6,     4,     1,     6,     6,     4,     5,
       6,     7,     8,     9,   577,   578,   579,   301,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,     3,
       3,    27,    28,     4,     3,     6,     3,   114,   115,     3,
       3,     3,     3,     3,   607,     3,   609,     3,   611,    45,
       1,    47,     3,    49,   620,   621,    52,    53,     3,    10,
      56,    57,    58,    57,    60,     3,     1,     1,     4,     6,
       3,     3,   624,    24,    25,   627,     6,     3,   630,     3,
       3,     1,    78,     3,     6,     6,    82,     3,     6,     6,
      10,     3,     6,    44,   655,   379,     3,    93,    94,    75,
      57,     3,    98,    62,    24,    25,     3,   693,    33,     3,
     106,   107,   108,   109,   110,   111,     6,   113,   114,   115,
     116,     3,     3,     3,    44,    10,     3,     6,     6,     3,
     693,     3,    30,     3,    10,   419,    55,   421,   422,   423,
     424,   425,   426,   427,   428,   429,   430,   431,   432,    10,
     434,   435,   436,   437,   438,   439,   440,   441,   442,   443,
      10,   445,   446,   729,    10,     3,   732,    56,    78,   735,
       3,     3,     6,     3,    77,   459,   460,     6,     6,   745,
       6,   465,     3,   467,     3,     3,   470,     3,     3,     3,
       3,     3,     3,     3,   746,     3,     3,   758,    55,    44,
      55,     3,     3,   766,     3,    78,   769,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,     6,   502,     6,
       3,     3,     3,   507,   508,     3,   510,     3,     3,   795,
       3,     3,   798,    55,    75,     3,     3,   803,     3,   805,
      57,     3,     3,     3,   796,     3,     3,   799,    57,     3,
     727,   529,   804,   405,   806,   743,   752,   547,   757,   514,
     590,   361,   590,   474,   695,   825,    -1,    31,   183,    -1,
     480,    -1,    -1,   839,    -1,    -1,   560,    -1,    -1,    -1,
      -1,   565,   566,   567,   568,   569,   570,   571,   572,   573,
     574,   575,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     0,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    -1,
      16,    -1,    -1,    19,    -1,    -1,    -1,    23,    24,    -1,
      26,    27,    28,    29,   618,    31,    32,    -1,    34,    35,
      36,    -1,    38,    39,    40,    41,    42,    43,    -1,    45,
      -1,    47,    48,    49,    50,    51,    52,    53,    54,    -1,
      56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,   671,    -1,   673,
      86,   675,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,   107,   108,   109,   110,   111,    -1,   113,   114,   115,
     116,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   713,
      -1,    -1,    -1,    -1,    -1,   719,   720,    -1,    -1,    -1,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    20,
      21,    22,    23,    -1,    -1,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,   813,
      -1,   815,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    44,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    -1,    -1,
      19,    -1,    -1,    -1,    23,    24,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    44,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    44,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    -1,    16,
      -1,    -1,    19,    -1,    -1,    -1,    23,    24,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      -1,    -1,    39,    -1,    -1,    -1,    -1,    44,    45,    -1,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,
      -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,    30,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,
      -1,    26,    27,    28,    29,    30,    31,    32,    -1,    34,
      35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    -1,    -1,
      19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    -1,    -1,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    -1,    16,
      -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,
      -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,
      -1,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    -1,    -1,
      19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    -1,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    -1,    -1,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    -1,    16,
      -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,
      -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      -1,    16,    -1,    -1,    19,    -1,    -1,    -1,    23,    -1,
      -1,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    -1,    16,    -1,    -1,
      19,    -1,    -1,    -1,    23,    -1,    -1,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    -1,    -1,
      39,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    -1,    16,    -1,    -1,    19,    -1,    -1,    -1,
      23,    -1,    -1,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    -1,    -1,    39,    -1,    -1,    -1,
      -1,    -1,    45,    -1,    47,    48,    49,    50,    -1,    52,
      53,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    -1,    16,
      -1,    -1,    19,    -1,    -1,    -1,    23,    -1,    -1,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    -1,
      11,    12,    13,    14,    -1,    16,    -1,    -1,    19,    -1,
      -1,    -1,    23,    -1,    -1,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    -1,    -1,    39,    -1,
      -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,   104,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,     9,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,     9,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,     1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,    -1,    44,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    56,    -1,
      58,    59,    -1,    -1,    -1,     1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    44,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   114,   115,    -1,    -1,
      56,    -1,    58,    59,    -1,    -1,    -1,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
       3,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   114,   115,
      -1,    44,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    56,    -1,    58,    59,    -1,    -1,    -1,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,     3,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   114,   115,    -1,    44,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     3,    56,    -1,    58,    59,
      -1,    -1,    -1,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,     3,    56,    -1,
      58,    59,    -1,    -1,   114,   115,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,     3,
      56,    -1,    58,    59,    -1,    -1,   114,   115,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,     3,    56,    -1,    58,    59,    -1,    -1,   114,   115,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,     3,    56,    -1,    58,    59,    -1,    -1,
     114,   115,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,     3,    56,    -1,    58,    59,
      -1,    -1,   114,   115,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,     3,    56,    -1,
      58,    59,    -1,    -1,   114,   115,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,     3,
      56,    -1,    58,    59,    -1,    -1,   114,   115,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,     3,    56,    -1,    58,    59,    -1,    -1,   114,   115,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    56,    -1,    58,    59,    -1,    -1,
     114,   115,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,     4,     5,     6,     7,     8,     9,
      -1,    -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    27,    28,    56,
      -1,    58,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    44,    45,    -1,    47,    -1,    49,
      -1,    -1,    52,    53,    -1,    -1,    56,    57,    58,    -1,
      60,    -1,    62,    90,    91,    92,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    -1,   114,   115,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,     4,
       5,     6,     7,     8,     9,    -1,   106,   107,   108,   109,
     110,   111,    -1,   113,   114,   115,   116,    -1,    -1,    -1,
      -1,    -1,    27,    28,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    44,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,   102,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    57,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     4,     5,     6,     7,     8,     9,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    27,    28,    -1,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    57,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,
       9,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    27,    28,
      -1,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,    -1,   113,   114,   115,   116,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    55,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,     9,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      27,    28,    -1,    -1,    -1,    -1,    -1,   106,   107,   108,
     109,   110,   111,    -1,   113,   114,   115,   116,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      57,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,     9,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    27,    28,    -1,    -1,    -1,    -1,    -1,   106,
     107,   108,   109,   110,   111,    -1,   113,   114,   115,   116,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      55,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,     9,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    27,    28,    -1,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,    -1,    44,   106,   107,   108,   109,   110,   111,    -1,
     113,   114,   115,   116,    56,    -1,    58,    59,    -1,    -1,
      62,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    44,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   114,   115,    -1,    -1,    56,    57,    58,    59,
      -1,    -1,    -1,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    44,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   114,   115,    -1,    -1,    56,    57,
      58,    59,    -1,    -1,    -1,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    55,
      56,    -1,    58,    59,    -1,    -1,   114,   115,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    56,    57,    58,    59,    -1,    -1,   114,   115,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    56,    -1,    58,    59,    -1,    -1,
     114,   115,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    77,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,    -1,    -1,    56,    -1,
      58,    59,   114,   115,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      56,    57,    58,    59,    -1,    -1,   114,   115,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    56,    57,    58,    59,    -1,    -1,   114,   115,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    56,    -1,    58,    59,    -1,    -1,
     114,   115,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    56,    -1,    58,    59,    -1,    -1,
      -1,    -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75,    -1,    -1,    78,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    56,    -1,    58,    59,    -1,    -1,
      -1,    -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    56,    -1,    58,    59,    -1,    -1,
      -1,    -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    56,    -1,    58,    59,    -1,    -1,
      -1,    -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    56,    -1,    58,    59,    -1,    -1,
      -1,    -1,   114,   115,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   114,   115
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   118,   119,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    16,    19,    23,
      24,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      38,    39,    40,    41,    42,    43,    45,    47,    48,    49,
      50,    51,    52,    53,    54,    56,    58,    60,    82,    86,
      93,    94,    98,   104,   106,   107,   108,   109,   110,   111,
     113,   114,   115,   116,   120,   121,   123,   124,   125,   127,
     128,   130,   131,   132,   135,   137,   138,   146,   147,   148,
     149,   155,   156,   157,   164,   166,   179,   181,   190,   192,
     199,   200,   201,   203,   205,   213,   214,   215,   217,   218,
     220,   223,   241,   246,   251,   255,   256,   258,   259,   261,
     262,   263,   265,   267,   271,   273,   274,   275,   276,   277,
       3,    44,    63,     3,     1,     6,   126,   258,     1,     6,
      45,   261,     1,     3,     1,     3,    15,     1,   261,     1,
     258,   280,     1,   261,     1,     1,   261,     1,     3,    44,
       1,   261,     1,     6,     1,     6,     1,     3,   261,   252,
       1,     6,     7,     1,   261,   263,     1,     6,     1,     3,
       6,   216,     1,     6,    33,   219,     1,     6,   221,   222,
       1,     6,   266,   272,     1,   261,    57,   261,   278,     1,
       3,    44,     6,   261,    44,    57,    62,   261,   277,   281,
     268,   261,     1,     3,   261,   277,   261,   261,   261,     1,
       3,   277,   261,   261,   261,   261,   261,   261,     4,     6,
      27,    59,   261,   261,     4,   258,   129,    32,    34,    45,
     124,   136,   124,     3,    44,   165,   180,   191,    46,   208,
     211,   212,   124,     1,    56,     3,    56,    58,    59,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    79,    80,    81,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   114,   115,   262,    75,    78,
       1,     4,     5,     7,     8,     9,    52,    53,    98,   122,
     257,   261,     3,     3,    78,    75,     3,    44,     3,    44,
       3,     3,     3,     3,    44,     3,    44,     3,    75,    78,
      92,     3,     3,     3,     3,     3,     3,   124,     3,     3,
       3,   224,     3,   247,     3,     3,     1,     6,   253,   254,
       3,     3,     3,     3,     3,     3,    75,     3,     3,    78,
       3,     1,     6,     7,     1,     3,    33,    78,     3,    75,
       3,    78,     3,     1,    56,   269,   269,     3,     3,     1,
      57,    78,   279,     3,   133,   124,   242,    55,    57,   261,
      57,    44,    62,     1,    57,     1,    57,    78,     1,     6,
     206,   207,   270,     3,     3,     3,     3,     4,     6,    27,
     145,   145,   150,   124,   167,   182,   145,     1,     3,    44,
     145,   209,   210,     3,     1,   206,    55,   277,   102,   261,
       6,   261,   261,   261,   261,   261,   261,   261,   261,   261,
     261,   261,   261,     1,   261,   261,   261,   261,   261,   261,
     261,   261,   261,   261,     6,   261,   261,     3,   260,   260,
     260,   260,   260,   260,   260,   260,   260,   260,   260,   261,
     261,     3,     4,     3,   258,   261,   151,   261,   258,     1,
     261,     1,    56,   225,   226,     1,    33,   228,   248,     3,
      78,     1,     1,   257,     6,     3,     3,    76,     3,    76,
       3,     6,     7,     6,     6,     7,   122,   222,     3,   206,
     208,   208,   261,   145,     3,    57,    57,   261,   261,    57,
     261,    62,     1,    62,    78,   208,    10,   124,    17,    18,
     139,   141,   142,   144,    20,    21,    22,   124,   153,   154,
     158,   160,   162,     1,     3,    24,    25,   168,   173,   175,
       1,     3,    24,   173,   183,    30,   193,   194,   195,   196,
       3,    44,    10,   145,   124,   204,     1,    55,     1,    55,
     261,    57,    78,     1,    44,   261,   261,   261,   261,   261,
     261,   261,   261,   261,   261,   261,     3,    78,    75,    77,
       3,     3,   206,   232,   228,     3,     6,   229,   230,     3,
     249,     1,   254,     3,     3,     6,     6,     3,    76,    92,
       3,    76,    92,     1,    55,   145,   145,    10,   243,    44,
      57,    62,   207,   145,     3,     1,     3,     1,   261,    10,
     140,   143,     1,     3,    44,     1,     3,    44,     1,     3,
      44,    10,   153,     3,     1,     6,     7,     8,   122,   177,
     178,     1,    10,   174,     3,     1,     4,     6,   188,   189,
      10,     1,     3,     4,     6,    92,   197,   198,    10,   195,
     145,     3,    10,    55,   202,     3,    44,   264,    57,   277,
       1,   261,   277,   261,     1,   261,     1,    55,     3,     6,
      10,    37,    45,    46,    58,   200,   218,   233,   234,   236,
     237,   238,     3,    56,   231,    78,     3,    10,   200,   218,
     234,   236,   250,     3,     3,     6,     6,     6,     6,     3,
      10,    10,   134,   261,     3,     6,    10,   218,   244,   261,
     261,    61,     3,     3,     3,     3,   145,   145,     3,   159,
     124,     3,   161,   124,     3,   163,   124,     3,     3,    44,
      77,     3,    44,    78,     3,     3,    44,   176,     3,    44,
       3,    44,    78,     3,     3,   258,     3,    78,    92,     3,
       3,    44,    55,    55,     3,     1,    78,   152,   227,    75,
       3,     3,     6,     6,    37,   239,    55,   277,   230,     3,
       3,     3,     3,     3,     3,     3,    75,    78,   245,     3,
      57,   139,   145,   145,   145,   171,   172,   122,   169,   170,
     178,   145,   124,   186,   187,   184,   185,   189,     3,   198,
     258,     3,     1,   261,    55,   261,   235,    75,    57,    57,
       3,    10,   200,   240,    55,   257,    10,    10,    10,   145,
     124,   145,   124,   145,   124,   145,   124,     3,     3,   208,
     257,     3,     3,     3,   245,     3,     3,     3,   145,     3,
      10,     3
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */





/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

    /* Number of syntax errors so far.  */
    int yynerrs;

    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:

/* Line 1455 of yacc.c  */
#line 207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:

/* Line 1455 of yacc.c  */
#line 212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat)=0; }
    break;

  case 10:

/* Line 1455 of yacc.c  */
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:

/* Line 1455 of yacc.c  */
#line 220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:

/* Line 1455 of yacc.c  */
#line 225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:

/* Line 1455 of yacc.c  */
#line 230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 14:

/* Line 1455 of yacc.c  */
#line 235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), false );
      }
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), true );
      }
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_stat)=0;}
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (! COMPILER->isInteractive()) && ((!(yyvsp[(1) - (2)].fal_val)->isExpr()) || (!(yyvsp[(1) - (2)].fal_val)->asExpr()->isStandAlone()) ) )
         {
            Falcon::StmtFunction *func = COMPILER->getFunctionContext();
            if ( func == 0 || ! func->isLambda() )
               COMPILER->raiseError(Falcon::e_noeffect );
         }

         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) );
      }
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (6)].fal_adecl)->size() != (yyvsp[(5) - (6)].fal_adecl)->size() + 1 )
         {
            COMPILER->raiseError(Falcon::e_unpack_size );
         }
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (6)].fal_adecl) );

         COMPILER->defineVal( first );
         (yyvsp[(5) - (6)].fal_adecl)->pushFront( (yyvsp[(3) - (6)].fal_val) );
         Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, second ) ) );
      }
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_def );
   }
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 349 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 351 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 378 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 387 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 394 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = static_cast<Falcon::StmtLoop* >(COMPILER->getContext());
         w->setCondition((yyvsp[(6) - (7)].fal_val));
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 402 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 408 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_loop );
      (yyval.fal_stat) = 0;
   }
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 415 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 416 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(1) - (1)].fal_val); }
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 420 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 428 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 435 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 446 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 473 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 491 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 498 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 507 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 515 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 525 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtForin( LINE, (yyvsp[(2) - (4)].fal_adecl), (yyvsp[(4) - (4)].fal_val) );
      }
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 547 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) );
         Falcon::ArrayDecl *decl = new Falcon::ArrayDecl();
         decl->pushBack( (yyvsp[(2) - (4)].fal_val) );
         (yyval.fal_stat) = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );  
      }
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 560 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 569 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>((yyvsp[(1) - (3)].fal_stat));
      if( (yyvsp[(3) - (3)].fal_stat) != 0 )
      {
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
         
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
      (yyval.fal_stat) = f;
   }
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 587 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>((yyvsp[(1) - (2)].fal_stat));
      
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 597 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 99:

/* Line 1455 of yacc.c  */
#line 608 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::RangeDecl* rd = new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val),
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(3) - (4)].fal_val))), (yyvsp[(4) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( rd );
      }
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 614 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val), 0 ) );
      }
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (3)].fal_val), 0, 0 ) );
      }
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 624 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 625 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 626 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 635 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 111:

/* Line 1455 of yacc.c  */
#line 649 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      }
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 662 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 670 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 674 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 680 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 686 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 693 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 698 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 707 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 716 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      }
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 728 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 739 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 743 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      }
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 755 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 756 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 765 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 769 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      }
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 783 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 785 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 794 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 132:

/* Line 1455 of yacc.c  */
#line 798 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 133:

/* Line 1455 of yacc.c  */
#line 806 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 134:

/* Line 1455 of yacc.c  */
#line 815 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 135:

/* Line 1455 of yacc.c  */
#line 817 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 138:

/* Line 1455 of yacc.c  */
#line 826 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 140:

/* Line 1455 of yacc.c  */
#line 832 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 142:

/* Line 1455 of yacc.c  */
#line 842 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 143:

/* Line 1455 of yacc.c  */
#line 850 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 854 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 146:

/* Line 1455 of yacc.c  */
#line 866 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 147:

/* Line 1455 of yacc.c  */
#line 876 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 149:

/* Line 1455 of yacc.c  */
#line 885 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      }
    break;

  case 153:

/* Line 1455 of yacc.c  */
#line 899 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 155:

/* Line 1455 of yacc.c  */
#line 903 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 158:

/* Line 1455 of yacc.c  */
#line 915 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 159:

/* Line 1455 of yacc.c  */
#line 924 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 160:

/* Line 1455 of yacc.c  */
#line 936 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 161:

/* Line 1455 of yacc.c  */
#line 947 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 958 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 978 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 164:

/* Line 1455 of yacc.c  */
#line 986 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 165:

/* Line 1455 of yacc.c  */
#line 995 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 166:

/* Line 1455 of yacc.c  */
#line 997 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 169:

/* Line 1455 of yacc.c  */
#line 1006 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 171:

/* Line 1455 of yacc.c  */
#line 1012 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 173:

/* Line 1455 of yacc.c  */
#line 1022 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 174:

/* Line 1455 of yacc.c  */
#line 1031 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 175:

/* Line 1455 of yacc.c  */
#line 1035 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 177:

/* Line 1455 of yacc.c  */
#line 1047 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 178:

/* Line 1455 of yacc.c  */
#line 1057 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 182:

/* Line 1455 of yacc.c  */
#line 1071 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 183:

/* Line 1455 of yacc.c  */
#line 1083 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 184:

/* Line 1455 of yacc.c  */
#line 1103 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 185:

/* Line 1455 of yacc.c  */
#line 1110 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 186:

/* Line 1455 of yacc.c  */
#line 1120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 188:

/* Line 1455 of yacc.c  */
#line 1129 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 194:

/* Line 1455 of yacc.c  */
#line 1149 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 195:

/* Line 1455 of yacc.c  */
#line 1167 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 1187 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 1197 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 1208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 201:

/* Line 1455 of yacc.c  */
#line 1221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 202:

/* Line 1455 of yacc.c  */
#line 1233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 1255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 1256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 1268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 1274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 208:

/* Line 1455 of yacc.c  */
#line 1283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 209:

/* Line 1455 of yacc.c  */
#line 1284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 210:

/* Line 1455 of yacc.c  */
#line 1287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 212:

/* Line 1455 of yacc.c  */
#line 1292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 213:

/* Line 1455 of yacc.c  */
#line 1293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 1300 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            func_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
         }
         else if ( parent != 0 && parent->type() == Falcon::Statement::t_state ) 
         {
            Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );
            func_name =  
                  stmt_state->owner()->symbol()->name() + "." + 
                  * stmt_state->name() + "#" + *(yyvsp[(2) - (2)].stringp);
         }
         else
            func_name = *(yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 )
         {
            if( parent->type() == Falcon::Statement::t_class ) 
            {
               Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
               Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
               if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
                  COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
               }
               else 
               {
                   cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
                   // is this a setter/getter?
                   if( ( (yyvsp[(2) - (2)].stringp)->find( "__set_" ) == 0 || (yyvsp[(2) - (2)].stringp)->find( "__get_" ) == 0 ) && (yyvsp[(2) - (2)].stringp)->length() > 6 )
                   {
                      Falcon::String *pname = COMPILER->addString( (yyvsp[(2) - (2)].stringp)->subString( 6 ));
                      Falcon::VarDef *pd = cd->getProperty( *pname );
                      if( pd == 0 )
                      {
                        pd = new Falcon::VarDef;
                        cd->addProperty( pname, pd );
                        pd->setReflective( Falcon::e_reflectSetGet, 0xFFFFFFFF );
                      }
                      else if( ! pd->isReflective() )
                      {
                        COMPILER->raiseError(Falcon::e_prop_adef, *pname );
                      }
   
                   }
               }
            }
            else if ( parent->type() == Falcon::Statement::t_state ) 
            {
   
               Falcon::StmtState *stmt_state = static_cast< Falcon::StmtState *>( parent );            
               if( ! stmt_state->addFunction( (yyvsp[(2) - (2)].stringp), sym ) )
               {
                  COMPILER->raiseError(Falcon::e_sm_adef, *(yyvsp[(2) - (2)].stringp) );
               }
               else 
               {
                  stmt_state->state()->addFunction( *(yyvsp[(2) - (2)].stringp), sym );
               }
               
               // eventually add a property where to store this thing
               Falcon::ClassDef *cd = stmt_state->owner()->symbol()->getClassDef();
               if ( ! cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) )
                  cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef );
            }
         }
         
         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 218:

/* Line 1455 of yacc.c  */
#line 1407 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), *(yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      }
    break;

  case 220:

/* Line 1455 of yacc.c  */
#line 1424 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 221:

/* Line 1455 of yacc.c  */
#line 1430 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 1435 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 1441 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 225:

/* Line 1455 of yacc.c  */
#line 1450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 227:

/* Line 1455 of yacc.c  */
#line 1455 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 228:

/* Line 1455 of yacc.c  */
#line 1465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 1468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 1477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 1487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 1492 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 1504 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 1513 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 1520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 1528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( *(yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 237:

/* Line 1455 of yacc.c  */
#line 1533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( *(yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 238:

/* Line 1455 of yacc.c  */
#line 1541 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 239:

/* Line 1455 of yacc.c  */
#line 1546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), *(yyvsp[(4) - (5)].stringp), "", false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 1551 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), *(yyvsp[(4) - (5)].stringp), "", true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 1556 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( *symName, *(yyvsp[(4) - (7)].stringp), *(yyvsp[(6) - (7)].stringp), false );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );

         (yyval.fal_stat) = 0;
      }
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 1576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( *symName, *(yyvsp[(4) - (7)].stringp), *(yyvsp[(6) - (7)].stringp), true );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 243:

/* Line 1455 of yacc.c  */
#line 1595 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), *(yyvsp[(4) - (7)].stringp), *(yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 244:

/* Line 1455 of yacc.c  */
#line 1600 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), *(yyvsp[(4) - (7)].stringp), *(yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:

/* Line 1455 of yacc.c  */
#line 1605 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 246:

/* Line 1455 of yacc.c  */
#line 1610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (4)].fal_genericList)->begin();
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            delete symName;
            li = li->next();
         }
         delete (yyvsp[(2) - (4)].fal_genericList);

         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 1624 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 248:

/* Line 1455 of yacc.c  */
#line 1629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 1634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 1639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 1644 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 1653 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addAttribute( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val), LINE );
     }
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 1658 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_attrdecl );
     }
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 1665 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 1671 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 1683 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 1688 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 260:

/* Line 1455 of yacc.c  */
#line 1701 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 261:

/* Line 1455 of yacc.c  */
#line 1705 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 1709 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 1722 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      }
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 1754 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls != 0 )
         {
            if ( cls->ctorFunction() == 0  )
            {
               Falcon::ClassDef *cd = cls->symbol()->getClassDef();
               if ( ! cd->inheritance().empty() )
               {
                  COMPILER->buildCtorFor( cls );
                  // COMPILER->addStatement( func ); should be done in buildCtorFor
                  // cls->ctorFunction( func ); idem
               }
            }
            COMPILER->popContext();
            //We didn't pushed a context set
            COMPILER->popFunction();
         }
      }
    break;

  case 266:

/* Line 1455 of yacc.c  */
#line 1788 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 269:

/* Line 1455 of yacc.c  */
#line 1796 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 270:

/* Line 1455 of yacc.c  */
#line 1797 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 275:

/* Line 1455 of yacc.c  */
#line 1814 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol(*(yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            cls->addInitExpression(
               new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_inherit,
                     new Falcon::Value( sym ), (yyvsp[(2) - (2)].fal_val) ) ) );
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
            delete (yyvsp[(2) - (2)].fal_val);
         }
      }
    break;

  case 276:

/* Line 1455 of yacc.c  */
#line 1837 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 1838 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 1840 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = (yyvsp[(2) - (3)].fal_adecl) == 0 ? 0 : new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
   }
    break;

  case 282:

/* Line 1455 of yacc.c  */
#line 1853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 283:

/* Line 1455 of yacc.c  */
#line 1856 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 284:

/* Line 1455 of yacc.c  */
#line 1876 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtState* ss = static_cast<Falcon::StmtState *>((yyvsp[(1) - (1)].fal_stat));
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );         
         if( ! cls->addState( static_cast<Falcon::StmtState *>((yyvsp[(1) - (1)].fal_stat)) ) )
         {
            const Falcon::String* name = ss->name();
            // we're not using the state we created, afater all.
            delete ss->state();
            delete (yyvsp[(1) - (1)].fal_stat);
            COMPILER->raiseError( Falcon::e_state_adef, *name ); 
         }
         else {
            // ownership passes on to the classdef
            cls->symbol()->getClassDef()->addState( ss->name(), ss->state() );
            cls->symbol()->getClassDef()->addProperty( ss->name(), 
               new Falcon::VarDef( ss->name() ) );
         }
      }
    break;

  case 287:

/* Line 1455 of yacc.c  */
#line 1900 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushFunctionContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }
    break;

  case 288:

/* Line 1455 of yacc.c  */
#line 1925 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
         COMPILER->popFunctionContext();
      }
    break;

  case 289:

/* Line 1455 of yacc.c  */
#line 1935 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( prop_name, def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   }
    break;

  case 290:

/* Line 1455 of yacc.c  */
#line 1957 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   }
    break;

  case 291:

/* Line 1455 of yacc.c  */
#line 1989 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { 
         (yyval.fal_stat) = COMPILER->getContext(); 
         COMPILER->popContext();
      }
    break;

  case 292:

/* Line 1455 of yacc.c  */
#line 1997 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
            
         COMPILER->pushContext( 
            new Falcon::StmtState( (yyvsp[(2) - (4)].stringp), cls ) ); 
      }
    break;

  case 293:

/* Line 1455 of yacc.c  */
#line 2005 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass* cls = 
            static_cast<Falcon::StmtClass*>( COMPILER->getContext() );
         
         Falcon::StmtState* state = new Falcon::StmtState( COMPILER->addString( "init" ), cls );
         cls->initState( state );
         
         COMPILER->pushContext( state ); 
      }
    break;

  case 297:

/* Line 1455 of yacc.c  */
#line 2026 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 298:

/* Line 1455 of yacc.c  */
#line 2037 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );
            sym->setEnum( true );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );

         COMPILER->resetEnum();
      }
    break;

  case 299:

/* Line 1455 of yacc.c  */
#line 2071 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 303:

/* Line 1455 of yacc.c  */
#line 2088 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 305:

/* Line 1455 of yacc.c  */
#line 2093 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 308:

/* Line 1455 of yacc.c  */
#line 2108 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( cl_name );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( *(yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         cls->singleton( sym );

         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
    break;

  case 309:

/* Line 1455 of yacc.c  */
#line 2148 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 311:

/* Line 1455 of yacc.c  */
#line 2176 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 315:

/* Line 1455 of yacc.c  */
#line 2188 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 316:

/* Line 1455 of yacc.c  */
#line 2191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 319:

/* Line 1455 of yacc.c  */
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 320:

/* Line 1455 of yacc.c  */
#line 2225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      }
    break;

  case 322:

/* Line 1455 of yacc.c  */
#line 2239 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 323:

/* Line 1455 of yacc.c  */
#line 2244 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 325:

/* Line 1455 of yacc.c  */
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 326:

/* Line 1455 of yacc.c  */
#line 2257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( *(yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 327:

/* Line 1455 of yacc.c  */
#line 2272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 328:

/* Line 1455 of yacc.c  */
#line 2273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 329:

/* Line 1455 of yacc.c  */
#line 2274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 330:

/* Line 1455 of yacc.c  */
#line 2282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 331:

/* Line 1455 of yacc.c  */
#line 2283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setUnbound(); }
    break;

  case 332:

/* Line 1455 of yacc.c  */
#line 2284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 333:

/* Line 1455 of yacc.c  */
#line 2285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 334:

/* Line 1455 of yacc.c  */
#line 2286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 335:

/* Line 1455 of yacc.c  */
#line 2287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 336:

/* Line 1455 of yacc.c  */
#line 2288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 337:

/* Line 1455 of yacc.c  */
#line 2292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 338:

/* Line 1455 of yacc.c  */
#line 2293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setUnbound(); }
    break;

  case 339:

/* Line 1455 of yacc.c  */
#line 2294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 340:

/* Line 1455 of yacc.c  */
#line 2295 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 341:

/* Line 1455 of yacc.c  */
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 342:

/* Line 1455 of yacc.c  */
#line 2297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 343:

/* Line 1455 of yacc.c  */
#line 2298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 344:

/* Line 1455 of yacc.c  */
#line 2303 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( *(yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     }
    break;

  case 346:

/* Line 1455 of yacc.c  */
#line 2321 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 347:

/* Line 1455 of yacc.c  */
#line 2322 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtFunction *sfunc = COMPILER->getFunctionContext();
      if ( sfunc == 0 ) {
         COMPILER->raiseError(Falcon::e_fself_outside, COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value();
      }
      else
      {
         (yyval.fal_val) = new Falcon::Value( sfunc->symbol() );
      }
   }
    break;

  case 352:

/* Line 1455 of yacc.c  */
#line 2350 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 353:

/* Line 1455 of yacc.c  */
#line 2351 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, "%d", (int)(yyvsp[(2) - (2)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 354:

/* Line 1455 of yacc.c  */
#line 2352 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 355:

/* Line 1455 of yacc.c  */
#line 2353 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyvsp[(3) - (3)].stringp)->prepend( "." ); (yyval.fal_val)->setLBind( (yyvsp[(3) - (3)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 356:

/* Line 1455 of yacc.c  */
#line 2354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, ".%d", (int)(yyvsp[(3) - (3)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 357:

/* Line 1455 of yacc.c  */
#line 2355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(".self") ); /* do not add the symbol to the compiler */ }
    break;

  case 358:

/* Line 1455 of yacc.c  */
#line 2356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 359:

/* Line 1455 of yacc.c  */
#line 2357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 360:

/* Line 1455 of yacc.c  */
#line 2358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            // is this an immediate string sum ?
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isString() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str += *(yyvsp[(4) - (4)].fal_val)->asString();
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.writeNumber( (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
         }
    break;

  case 361:

/* Line 1455 of yacc.c  */
#line 2384 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 362:

/* Line 1455 of yacc.c  */
#line 2385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( (yyvsp[(1) - (4)].fal_val)->asString()->length() );
                  for( int i = 0; i < (yyvsp[(4) - (4)].fal_val)->asInteger(); ++i )
                  {
                     str.append( *(yyvsp[(1) - (4)].fal_val)->asString()  );
                  }
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 363:

/* Line 1455 of yacc.c  */
#line 2405 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if( (yyvsp[(1) - (4)].fal_val)->asString()->length() == 0 )
               {
                  COMPILER->raiseError( Falcon::e_invop );
               }
               else {

                  if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
                  {
                     Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                     str.setCharAt( str.length()-1, str.getCharAt(str.length()-1) + (yyvsp[(4) - (4)].fal_val)->asInteger() );
                     (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                     delete (yyvsp[(4) - (4)].fal_val);
                     (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
                  }
                  else
                     (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
               }
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 364:

/* Line 1455 of yacc.c  */
#line 2429 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.append( (Falcon::uint32) (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 365:

/* Line 1455 of yacc.c  */
#line 2446 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 366:

/* Line 1455 of yacc.c  */
#line 2447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 367:

/* Line 1455 of yacc.c  */
#line 2448 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 368:

/* Line 1455 of yacc.c  */
#line 2449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 369:

/* Line 1455 of yacc.c  */
#line 2450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 370:

/* Line 1455 of yacc.c  */
#line 2451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 371:

/* Line 1455 of yacc.c  */
#line 2452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 372:

/* Line 1455 of yacc.c  */
#line 2453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 373:

/* Line 1455 of yacc.c  */
#line 2454 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 374:

/* Line 1455 of yacc.c  */
#line 2455 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 375:

/* Line 1455 of yacc.c  */
#line 2456 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 376:

/* Line 1455 of yacc.c  */
#line 2457 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 377:

/* Line 1455 of yacc.c  */
#line 2458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 378:

/* Line 1455 of yacc.c  */
#line 2459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_exeq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 379:

/* Line 1455 of yacc.c  */
#line 2460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 380:

/* Line 1455 of yacc.c  */
#line 2461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 381:

/* Line 1455 of yacc.c  */
#line 2462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 382:

/* Line 1455 of yacc.c  */
#line 2463 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 383:

/* Line 1455 of yacc.c  */
#line 2464 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 384:

/* Line 1455 of yacc.c  */
#line 2465 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 385:

/* Line 1455 of yacc.c  */
#line 2466 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 386:

/* Line 1455 of yacc.c  */
#line 2467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 387:

/* Line 1455 of yacc.c  */
#line 2468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 388:

/* Line 1455 of yacc.c  */
#line 2469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 389:

/* Line 1455 of yacc.c  */
#line 2470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 390:

/* Line 1455 of yacc.c  */
#line 2471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 391:

/* Line 1455 of yacc.c  */
#line 2472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 392:

/* Line 1455 of yacc.c  */
#line 2473 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 393:

/* Line 1455 of yacc.c  */
#line 2474 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eval, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 394:

/* Line 1455 of yacc.c  */
#line 2475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 395:

/* Line 1455 of yacc.c  */
#line 2476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_deoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 396:

/* Line 1455 of yacc.c  */
#line 2477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_isoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 397:

/* Line 1455 of yacc.c  */
#line 2478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_xoroob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 404:

/* Line 1455 of yacc.c  */
#line 2486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 405:

/* Line 1455 of yacc.c  */
#line 2491 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 406:

/* Line 1455 of yacc.c  */
#line 2495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 407:

/* Line 1455 of yacc.c  */
#line 2500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 408:

/* Line 1455 of yacc.c  */
#line 2506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 411:

/* Line 1455 of yacc.c  */
#line 2518 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 412:

/* Line 1455 of yacc.c  */
#line 2523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 413:

/* Line 1455 of yacc.c  */
#line 2530 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 414:

/* Line 1455 of yacc.c  */
#line 2531 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 415:

/* Line 1455 of yacc.c  */
#line 2532 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 416:

/* Line 1455 of yacc.c  */
#line 2533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 417:

/* Line 1455 of yacc.c  */
#line 2534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 418:

/* Line 1455 of yacc.c  */
#line 2535 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 419:

/* Line 1455 of yacc.c  */
#line 2536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 420:

/* Line 1455 of yacc.c  */
#line 2537 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 421:

/* Line 1455 of yacc.c  */
#line 2538 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 422:

/* Line 1455 of yacc.c  */
#line 2539 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 423:

/* Line 1455 of yacc.c  */
#line 2540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 424:

/* Line 1455 of yacc.c  */
#line 2541 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 425:

/* Line 1455 of yacc.c  */
#line 2546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 426:

/* Line 1455 of yacc.c  */
#line 2549 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 427:

/* Line 1455 of yacc.c  */
#line 2552 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 428:

/* Line 1455 of yacc.c  */
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 429:

/* Line 1455 of yacc.c  */
#line 2558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 430:

/* Line 1455 of yacc.c  */
#line 2565 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 431:

/* Line 1455 of yacc.c  */
#line 2571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 432:

/* Line 1455 of yacc.c  */
#line 2575 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 433:

/* Line 1455 of yacc.c  */
#line 2576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 434:

/* Line 1455 of yacc.c  */
#line 2585 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );

         // Not defined?
         fassert( COMPILER->searchGlobalSymbol( name ) == 0 );
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 435:

/* Line 1455 of yacc.c  */
#line 2619 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 436:

/* Line 1455 of yacc.c  */
#line 2627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 437:

/* Line 1455 of yacc.c  */
#line 2661 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StatementList *stmt = COMPILER->getContextSet();
         if( stmt->size() == 1 && stmt->back()->type() == Falcon::Statement::t_autoexp )
         {
            // wrap it around a return, so A is not nilled.
            Falcon::StmtAutoexpr *ae = static_cast<Falcon::StmtAutoexpr *>( stmt->pop_back() );
            stmt->push_back( new Falcon::StmtReturn( 1, ae->value()->clone() ) );

            // we don't need the expression anymore.
            delete ae;
         }

         (yyval.fal_val) = COMPILER->closeClosure();
      }
    break;

  case 439:

/* Line 1455 of yacc.c  */
#line 2680 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 440:

/* Line 1455 of yacc.c  */
#line 2684 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 442:

/* Line 1455 of yacc.c  */
#line 2692 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 443:

/* Line 1455 of yacc.c  */
#line 2696 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 444:

/* Line 1455 of yacc.c  */
#line 2703 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String name( buf, -1 );

         // Not defined?
         fassert( COMPILER->searchGlobalSymbol( name ) == 0 );
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 445:

/* Line 1455 of yacc.c  */
#line 2736 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 446:

/* Line 1455 of yacc.c  */
#line 2752 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 447:

/* Line 1455 of yacc.c  */
#line 2757 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 448:

/* Line 1455 of yacc.c  */
#line 2764 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 449:

/* Line 1455 of yacc.c  */
#line 2771 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 450:

/* Line 1455 of yacc.c  */
#line 2780 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 451:

/* Line 1455 of yacc.c  */
#line 2782 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 452:

/* Line 1455 of yacc.c  */
#line 2786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 453:

/* Line 1455 of yacc.c  */
#line 2793 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 454:

/* Line 1455 of yacc.c  */
#line 2795 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 455:

/* Line 1455 of yacc.c  */
#line 2799 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 456:

/* Line 1455 of yacc.c  */
#line 2807 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 457:

/* Line 1455 of yacc.c  */
#line 2808 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 458:

/* Line 1455 of yacc.c  */
#line 2810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 459:

/* Line 1455 of yacc.c  */
#line 2817 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 460:

/* Line 1455 of yacc.c  */
#line 2818 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 461:

/* Line 1455 of yacc.c  */
#line 2822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 462:

/* Line 1455 of yacc.c  */
#line 2823 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 465:

/* Line 1455 of yacc.c  */
#line 2830 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 466:

/* Line 1455 of yacc.c  */
#line 2836 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 467:

/* Line 1455 of yacc.c  */
#line 2843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 468:

/* Line 1455 of yacc.c  */
#line 2844 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;



/* Line 1455 of yacc.c  */
#line 7540 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 1675 of yacc.c  */
#line 2848 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


